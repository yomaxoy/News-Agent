import os
import yaml
import feedparser
import requests
from openai import OpenAI
from datetime import datetime, timezone, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def send_discord(text, webhook_url, username="News Agent"):
    """
    Sendet Text über den Discord Webhook. Da Discord ein Limit von
    2000 Zeichen pro Nachricht hat, wird der Text intelligent an
    Zeilenumbrüchen aufgeteilt.
    """
    chunks = []
    current_chunk = ""
    for line in text.split("\n"):
        if len(current_chunk) + len(line) + 1 > 1900:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += "\n" + line if current_chunk else line
    if current_chunk:
        chunks.append(current_chunk)

    for i, chunk in enumerate(chunks):
        response = requests.post(webhook_url, json={
            "content": chunk,
            "username": username,
        })
        if response.status_code == 204:
            print(f"Nachricht {i+1}/{len(chunks)} gesendet")
        else:
            print(f"Fehler: {response.status_code} - {response.text}")


def notify_error(error_message):
    """
    Sendet eine Fehlermeldung an Discord, falls der Agent abstürzt.
    So bekommst du immer mit, wenn ein Lauf fehlschlägt.
    """
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return
    send_discord(
        f"**News Agent Fehler** ({datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M UTC')})\n"
        f"```\n{error_message}\n```\nBitte GitHub Actions Logs prüfen.",
        webhook_url,
        username="News Agent [Fehler]",
    )


def fetch_articles(sources, limit=60, max_age_hours=24):
    """
    Geht alle konfigurierten RSS-Feeds durch und sammelt
    Titel, Zusammenfassung, Link und Quelle jedes Artikels.
    Nur Artikel der letzten `max_age_hours` Stunden werden berücksichtigt.
    Fehler bei einzelnen Feeds werden abgefangen, damit ein
    kaputter Feed nicht den gesamten Agent zum Absturz bringt.
    """
    articles = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

    for source in sources:
        try:
            feed = feedparser.parse(source["url"])
            for entry in feed.entries[:15]:
                # Freshness-Check: Artikel ohne Datum oder zu alte Artikel überspringen
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if published:
                    pub_dt = datetime(*published[:6], tzinfo=timezone.utc)
                    if pub_dt < cutoff:
                        continue

                articles.append({
                    "title": entry.get("title", "Kein Titel"),
                    "summary": entry.get("summary", "")[:300],
                    "link": entry.get("link", ""),
                    "source": source.get("name", feed.feed.get("title", "Unbekannt")),
                })
        except Exception as e:
            print(f"Fehler beim Laden von {source['name']}: {e}")

    print(f"{len(articles)} Artikel (letzte {max_age_hours}h) aus {len(sources)} Quellen geladen")
    return articles[:limit]


def deduplicate(articles):
    """
    Entfernt Artikel mit sehr ähnlichen Titeln, da verschiedene
    Feeds oft über dasselbe Thema berichten. Nutzt einfache
    Wort-Überlappung als Ähnlichkeitsmaß.
    """
    seen_titles = []
    unique = []
    for article in articles:
        title_words = set(article["title"].lower().split())
        is_duplicate = False
        for seen in seen_titles:
            overlap = len(title_words & seen) / max(len(title_words | seen), 1)
            if overlap > 0.6:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(article)
            seen_titles.append(title_words)

    removed = len(articles) - len(unique)
    if removed > 0:
        print(f"{removed} Duplikate entfernt")
    return unique


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
def generate_digest(articles, config):
    """
    Sendet alle Artikel an Groq (Llama 3.3 70B) zusammen mit dem
    Interessenprofil. Das Modell wählt die relevantesten Artikel
    aus und fasst sie zusammen. Die niedrige Temperature (0.3)
    sorgt für faktengetreue Zusammenfassungen.
    Bei API-Fehlern oder Rate Limits wird der Call automatisch bis
    zu 3 Mal mit exponentiellem Backoff wiederholt (tenacity).
    """
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )

    today = datetime.now(timezone.utc).strftime("%d.%m.%Y")

    articles_text = "\n\n".join(
        f"[{a['source']}] {a['title']}\n{a['summary']}\nURL: {a['link']}"
        for a in articles
    )

    prompt = f"""Du bist mein persönlicher News-Kurator. Heute ist der {today}.

Mein Interessenprofil: {config['profile']}

Hier sind die heutigen Artikel aus verschiedenen Quellen:

{articles_text}

---
AUFGABE:
1. Wähle die {config['max_articles']} relevantesten Artikel für mein Profil aus
2. Fasse jeden Artikel in 2-3 Sätzen auf {config['language']} zusammen
3. Erkläre in einem Satz, warum der Artikel für mich relevant ist
4. Füge den Original-Link an

FORMAT:
Erstelle einen übersichtlichen Daily Digest für Discord.
Nutze **fett** für Artikeltitel.
Setze Links in < > damit Discord keine Preview generiert.
Nutze --- als Trenner zwischen Artikeln.
Beginne mit einer kurzen Begrüssung und dem Datum.
Am Ende: Ein kurzes Fazit mit dem wichtigsten Trend des Tages."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2048,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("News Agent gestartet...")

    try:
        config = load_config()
        articles = fetch_articles(
            config["sources"],
            limit=config.get("max_article_fetch", 60),
            max_age_hours=config.get("article_max_age_hours", 24),
        )

        if not articles:
            raise RuntimeError("Keine Artikel der letzten 24h gefunden. Möglicherweise sind alle Feeds leer oder offline.")

        articles = deduplicate(articles)
        digest = generate_digest(articles, config)
        print("\n" + digest + "\n")

        if os.environ.get("DISCORD_WEBHOOK_URL"):
            send_discord(digest, os.environ["DISCORD_WEBHOOK_URL"])
        else:
            print("Kein DISCORD_WEBHOOK_URL gesetzt - nur Konsolenausgabe")

        print("Fertig!")

    except Exception as e:
        print(f"FEHLER: {e}")
        notify_error(str(e))
        raise
