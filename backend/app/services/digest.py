"""Digest generation service"""
from sqlalchemy.orm import Session
from app.db.models import Source, Article, Digest, Schedule
from app.integrations.groq import GroqClient
from app.services.source import SourceService
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import hashlib
import logging
import feedparser

logger = logging.getLogger(__name__)

class DigestService:
    @staticmethod
    def fetch_articles_from_sources(
        db: Session,
        source_ids: List[int],
        max_articles: int = 20,
        max_age_hours: int = 24
    ) -> List[Article]:
        """Fetch articles from specified sources"""
        articles = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        for source_id in source_ids:
            try:
                source = db.query(Source).filter(Source.id == source_id).first()
                if not source:
                    continue

                feed_result = SourceService.test_feed(source.url)
                if not feed_result.valid:
                    logger.warning(f"Feed {source.name} is invalid: {feed_result.error}")
                    continue

                # Parse feed and create articles
                feed = feedparser.parse(source.url)

                for entry in feed.entries[:15]:
                    # Get article date
                    published = entry.get("published_parsed") or entry.get("updated_parsed")
                    if published:
                        pub_dt = datetime(*published[:6], tzinfo=timezone.utc)
                        if pub_dt < cutoff:
                            continue

                    # Check if article already exists
                    external_id = entry.get("id", entry.get("link", ""))
                    url = entry.get("link", "")

                    existing = db.query(Article).filter(
                        (Article.url == url) | (Article.external_id == external_id)
                    ).first()

                    if existing:
                        articles.append(existing)
                        continue

                    # Create new article
                    article = Article(
                        source_id=source.id,
                        external_id=external_id,
                        title=entry.get("title", "No Title")[:500],
                        summary=entry.get("summary", "")[:1000],
                        url=url,
                        published_at=datetime(*published[:6], tzinfo=timezone.utc) if published else datetime.now(timezone.utc)
                    )
                    db.add(article)
                    articles.append(article)

                db.commit()

            except Exception as e:
                logger.error(f"Error fetching from source {source_id}: {e}")
                continue

        logger.info(f"Fetched {len(articles)} articles from {len(source_ids)} sources")
        return articles[:max_articles]

    @staticmethod
    def deduplicate_articles(articles: List[Article]) -> List[Article]:
        """Remove duplicate articles based on title similarity"""
        if not articles:
            return []

        unique = []
        seen_hashes = set()
        seen_titles = []

        for article in articles:
            # Quick hash-based deduplication
            title_hash = hashlib.md5(f"{article.title}:{article.source_id}".encode()).hexdigest()
            if title_hash in seen_hashes:
                continue

            # Word overlap deduplication
            title_words = set(article.title.lower().split())
            is_duplicate = False

            for seen in seen_titles:
                overlap = len(title_words & seen) / max(len(title_words | seen), 1)
                if overlap > 0.7:  # Increased threshold from 0.6 for better accuracy
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(article)
                seen_hashes.add(title_hash)
                seen_titles.append(title_words)

        removed = len(articles) - len(unique)
        if removed > 0:
            logger.info(f"Removed {removed} duplicates")

        return unique

    @staticmethod
    def generate_digest(
        articles: List[Article],
        profile: str = "Technology and AI news",
        language: str = "English",
        max_articles: int = 5
    ) -> str:
        """Generate digest using Groq API"""
        if not articles:
            return "No articles available to generate digest."

        try:
            groq = GroqClient()
        except ValueError as e:
            logger.error(f"Groq not configured: {e}")
            return DigestService._fallback_digest(articles)

        # Format articles for Groq
        articles_text = "\n\n".join(
            f"[{article.source.name}] {article.title}\n"
            f"{article.summary}\n"
            f"URL: <{article.url}>"
            for article in articles
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        prompt = f"""You are a personal news curator. Today is {today}.

My Interest Profile: {profile}

Here are today's articles from various sources:

{articles_text}

---
TASK:
1. Select the {max_articles} most relevant articles for my profile
2. Summarize each article in 2-3 sentences in {language}
3. Explain in one sentence why the article is relevant to me
4. Include the original link

FORMAT:
Create a well-structured daily digest.
Use **bold** for article titles.
Use --- as separator between articles.
Begin with a brief greeting and today's date.
End with a brief summary of the most important trend today."""

        try:
            digest = groq.create_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048
            )
            return digest
        except Exception as e:
            logger.error(f"Error generating digest with Groq: {e}")
            return DigestService._fallback_digest(articles)

    @staticmethod
    def _fallback_digest(articles: List[Article]) -> str:
        """Fallback digest generation when Groq is unavailable"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        lines = [f"Daily News Digest – {today}\n", "=" * 50]

        if not articles:
            lines.append("\n⚠️ No articles available for this digest.")
            lines.append("\nPlease check your sources are active and have recent content.")
        else:
            for article in articles[:5]:
                lines.append(f"\n**{article.title}**")
                lines.append(f"Source: {article.source.name}")
                if article.summary:
                    lines.append(article.summary[:300])
                else:
                    lines.append("(No summary available)")
                if article.url:
                    lines.append(f"Read more: {article.url}")
                lines.append("---")

        digest_text = "\n".join(lines)
        # Ensure we never return empty string
        return digest_text if digest_text.strip() else "Daily News Digest Generated"

    @staticmethod
    def save_digest(
        db: Session,
        schedule_id: int,
        content: str,
        content_format: str = "markdown"
    ) -> Digest:
        """Save generated digest to database"""
        digest = Digest(
            schedule_id=schedule_id,
            content_text=content if content_format == "text" else content,
            content_html=content if content_format == "html" else None,
            status="generated"
        )
        db.add(digest)
        db.commit()
        db.refresh(digest)
        return digest
