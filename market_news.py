# market_news.py
import streamlit as st
import feedparser
from html import escape

# ----------------------
# RSS Feed Sources
# ----------------------
FEEDS = {
    "stocks": "https://www.investing.com/rss/news_25.rss",
    "gold": "https://www.investing.com/rss/commodities_news.rss",
    "dollar": "https://www.investing.com/rss/currencies_news.rss",
    "metals": "https://www.investing.com/rss/commodities_news.rss"
}

# ----------------------
# Fetch news function
# ----------------------
def fetch_news(category="stocks", limit=10):
    """
    Fetch latest news from RSS feed based on category.
    Returns a list of dictionaries with title, link, published date, and image.
    """
    feed_url = FEEDS.get(category)
    if not feed_url:
        return []

    feed = feedparser.parse(feed_url)
    news_items = []
    for entry in feed.entries[:limit]:
        # Try to get image if available
        image = None
        if 'media_content' in entry:
            image = entry.media_content[0]['url']
        elif 'links' in entry:
            for link in entry.links:
                if link.type.startswith('image'):
                    image = link.href
                    break

        # Escape special characters in title/link
        title = escape(entry.title)
        link_url = escape(entry.link)

        news_items.append({
            "title": title,
            "link": link_url,
            "published": entry.get("published", "N/A"),
            "image": image,
            "sentiment": "Neutral"  # Placeholder for AI sentiment analysis
        })
    return news_items

# ----------------------
# Display news with images, animations, and placeholder text
# ----------------------
def display_news(category="stocks", limit=10):
    news_items = fetch_news(category, limit)
    
    st.markdown(f"### 📰 Latest {category.capitalize()} News")
    
    # Custom CSS for animated cards
    st.markdown(
        """
        <style>
        .news-card {
            display: flex;
            flex-direction: row;
            border: 1px solid #ccc;
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 12px;
            background-color: #fefefe;
            transition: all 0.4s ease-in-out;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .news-card:hover {
            transform: scale(1.03) rotateZ(0.5deg);
            box-shadow: 6px 8px 15px rgba(0,0,0,0.25);
            border: 1px solid #1a73e8;
            background-color: #f0f8ff;
        }
        .news-image {
            width: 100px;
            height: 70px;
            object-fit: cover;
            border-radius: 8px;
            margin-right: 12px;
        }
        .news-placeholder {
            width: 100px;
            height: 70px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            background-color: #e0e0e0;
            color: #555;
            font-size: 12px;
            margin-right: 12px;
            text-align: center;
        }
        .news-content {
            flex: 1;
        }
        .news-title {
            margin: 0;
            font-size: 16px;
            font-weight: bold;
        }
        .news-title a {
            text-decoration: none;
            color: #1a73e8;
        }
        .news-title a:hover {
            color: #d93025;
            text-decoration: underline;
        }
        .news-date {
            font-size: 12px;
            color: gray;
        }
        .news-sentiment {
            font-size: 12px;
            color: #444;
            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Display each news item
    for i, news in enumerate(news_items):
        if news["image"]:
            img_html = f'<img src="{news["image"]}" class="news-image">'
        else:
            img_html = '<div class="news-placeholder">No Image Available</div>'

        st.markdown(
            f"""
            <div class="news-card">
                {img_html}
                <div class="news-content">
                    <div class="news-title"><a href="{news['link']}" target="_blank">{news['title']}</a></div>
                    <div class="news-date">{news['published']}</div>
                    <div class="news-sentiment"><b>Sentiment:</b> {news['sentiment']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Optional button per card with unique key
        # st.button("Read More", key=f"readmore_{category}_{i}")
