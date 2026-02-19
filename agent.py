import io
import os
import struct
import wave
import yaml
import feedparser
import requests
from openai import OpenAI
from datetime import datetime, timezone, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.cloud import texttospeech


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


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
def generate_podcast_script(digest, config):
    """
    Wandelt den fertigen Digest in ein natürliches Zwei-Personen-Dialogskript um.
    Host (Alex) und Guest (Sara) wechseln sich ab und kommentieren die News.
    Das Format ist speziell auf Google Cloud TTS mit zwei Stimmen ausgelegt.
    Jede Zeile beginnt mit "ALEX:" oder "SARA:" als Sprecher-Marker.
    """
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )

    today = datetime.now(timezone.utc).strftime("%d.%m.%Y")

    prompt = f"""Du bist Autor eines täglichen Tech-News-Podcasts auf Deutsch. Heute ist der {today}.

Hier ist der heutige News-Digest:

{digest}

---
AUFGABE:
Schreibe ein natürliches Gesprächsskript für zwei Moderatoren:
- ALEX: Der Haupt-Moderator, erklärt die News sachlich und strukturiert
- SARA: Die Co-Moderatorin, stellt kluge Nachfragen, liefert Kontext und Einordnung

REGELN:
- Jede Zeile beginnt mit genau "ALEX:" oder "SARA:" (kein anderes Format)
- Natürliche Sprache, keine Bulletpoints oder Markdown
- Kurze, verständliche Sätze (Podcast-Stil, nicht Vorlesungsstil)
- Decke alle Artikel aus dem Digest ab
- Intro: Alex begrüßt, Sara steigt mit erstem Thema ein
- Outro: gemeinsames kurzes Fazit, Verabschiedung
- Ziel: ca. 5 Minuten Sprechzeit (ca. 750 Wörter gesamt)

Beginne direkt mit dem Skript, ohne Präambel."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=2048,
    )

    return response.choices[0].message.content


def generate_podcast_audio(script, config, output_path):
    """
    Synthetisiert das Podcast-Skript mit Google Cloud TTS (WaveNet).
    Jede Zeile wird mit der passenden Stimme (ALEX/SARA) vertont und
    die Audio-Segmente werden zu einer einzigen MP3-Datei zusammengefügt.
    Authentifizierung erfolgt über GOOGLE_APPLICATION_CREDENTIALS (Service Account JSON).
    """
    podcast_cfg = config.get("podcast", {})
    voice_host = podcast_cfg.get("voice_host", "de-DE-Wavenet-B")
    voice_guest = podcast_cfg.get("voice_guest", "de-DE-Wavenet-A")
    speaking_rate = podcast_cfg.get("speaking_rate", 1.05)

    client = texttospeech.TextToSpeechClient()

    audio_segments = []
    lines = [line.strip() for line in script.strip().split("\n") if line.strip()]

    for line in lines:
        if line.startswith("ALEX:"):
            text = line[5:].strip()
            voice_name = voice_host
        elif line.startswith("SARA:"):
            text = line[5:].strip()
            voice_name = voice_guest
        else:
            # Zeilen ohne Sprecher-Marker überspringen
            continue

        if not text:
            continue

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="de-DE",
            name=voice_name,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        audio_segments.append(response.audio_content)

    # Alle MP3-Segmente zu einer Datei zusammenfügen
    with open(output_path, "wb") as f:
        for segment in audio_segments:
            f.write(segment)

    size_kb = os.path.getsize(output_path) // 1024
    print(f"Podcast gespeichert: {output_path} ({size_kb} KB, {len(audio_segments)} Segmente)")
    return output_path


def send_discord_file(file_path, webhook_url, message=""):
    """
    Lädt eine Datei (z.B. MP3) direkt als Anhang in Discord hoch.
    Discord erlaubt Dateianhänge bis 8 MB über Webhooks.
    """
    with open(file_path, "rb") as f:
        response = requests.post(
            webhook_url,
            data={"content": message, "username": "News Agent"},
            files={"file": (os.path.basename(file_path), f, "audio/mpeg")},
        )
    if response.status_code == 200:
        print(f"Podcast-Datei erfolgreich in Discord gepostet")
    else:
        print(f"Fehler beim Hochladen: {response.status_code} - {response.text}")


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

        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
        if webhook_url:
            send_discord(digest, webhook_url)
        else:
            print("Kein DISCORD_WEBHOOK_URL gesetzt - nur Konsolenausgabe")

        # Podcast generieren (optional, nur wenn aktiviert und Credentials vorhanden)
        podcast_cfg = config.get("podcast", {})
        if podcast_cfg.get("enabled") and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            print("Generiere Podcast-Skript...")
            script = generate_podcast_script(digest, config)
            print("Synthetisiere Audio mit Google Cloud TTS...")
            output_file = podcast_cfg.get("output_file", "podcast.mp3")
            generate_podcast_audio(script, config, output_file)
            if webhook_url:
                today = datetime.now(timezone.utc).strftime("%d.%m.%Y")
                send_discord_file(
                    output_file,
                    webhook_url,
                    message=f"**Daily News Podcast – {today}**",
                )
        elif podcast_cfg.get("enabled"):
            print("Podcast deaktiviert: GOOGLE_APPLICATION_CREDENTIALS nicht gesetzt")

        print("Fertig!")

    except Exception as e:
        print(f"FEHLER: {e}")
        notify_error(str(e))
        raise
