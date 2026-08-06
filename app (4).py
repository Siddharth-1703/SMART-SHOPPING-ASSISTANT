# ==============================
# SMART SHOPPING ASSISTANT
# Retail & E-Commerce using GenAI
# ==============================
import re
import warnings
import requests
import streamlit as st
import streamlit.components.v1 as components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tavily import TavilyClient
warnings.filterwarnings("ignore")
# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="Smart Shopping Assistant",
    page_icon="🛍️",
    layout="wide"
)
st.title("🛒 Smart Shopping Recommendation Assistant")
st.write("""
This AI Assistant recommends products based on
✔ Budget
✔ Requirements
✔ Product Features
✔ Comparison
✔ AI Buying Suggestion
✔ Alternatives
✔ Latest Product Search
""")
st.divider()
# ==============================
# SIDEBAR - API KEYS
# ==============================
st.sidebar.title("API Configuration")
GOOGLE_API_KEY = st.sidebar.text_input("Gemini API Key", type="password")
TAVILY_API_KEY = st.sidebar.text_input("Tavily API Key", type="password")

if not all([GOOGLE_API_KEY, TAVILY_API_KEY]):
    st.warning("Please Enter All API Keys")
    st.stop()

st.sidebar.success("API Loaded Successfully")

# ==============================
# SIDEBAR - ABOUT PROJECT
# ==============================
with st.sidebar.expander("ℹ️ About This Project", expanded=False):
    st.markdown("""
**🛍️ Smart Shopping Recommendation Assistant**

An AI-powered shopping assistant built with:
- **Streamlit** – interactive UI
- **Google Gemini** – recommendations, comparisons & buying verdicts
- **Tavily Search** – live web search for real Amazon/Flipkart
  product links, photos and reviews
- **FakeStore API** – sample product catalog

**Features**
- Budget, brand & rating based filtering
- AI product comparison & recommendation
- Alternatives suggestion
- Live buy links with price in **USD ($)** and **INR (₹)**
- Latest product reviews pulled from the web
""")

# ==============================
# GEMINI MODEL
# ==============================
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY
)
# ==============================
# SHOPPING DETAILS
# ==============================
st.sidebar.title("Shopping Preferences")
category = st.sidebar.selectbox(
    "Product Category",
    ["electronics", "jewelery", "men's clothing", "women's clothing"]
)
budget = st.sidebar.number_input(
    "Budget",
    min_value=100,
    max_value=1000000,
    value=50000
)
brand = st.sidebar.text_input("Preferred Brand", placeholder="Optional")
rating = st.sidebar.slider("Minimum Rating", 1, 5, 4)
st.markdown("## Product Requirements")
requirements = st.text_area(
    "Describe your requirements",
    height=180,
    placeholder="""Example:
Gaming Laptop
RTX 4060
16GB RAM
1TB SSD
Battery Backup
Video Editing
Budget 70000
"""
)
st.divider()

# ===========================================
# CURRENCY: LIVE USD -> INR RATE
# ===========================================
@st.cache_data(ttl=3600)
def get_usd_to_inr_rate() -> float:
    """Fetch live USD->INR rate; fall back to a fixed approximate rate."""
    try:
        resp = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=10)
        if resp.status_code == 200:
            rate = resp.json().get("rates", {}).get("INR")
            if rate:
                return float(rate)
    except requests.RequestException:
        pass
    return 83.0  # fallback approximate rate

USD_TO_INR = get_usd_to_inr_rate()

def format_price(usd_price) -> str:
    """Return price formatted as both USD and INR."""
    try:
        usd_price = float(usd_price)
    except (TypeError, ValueError):
        return "N/A"
    inr_price = usd_price * USD_TO_INR
    return f"${usd_price:,.2f}  |  ₹{inr_price:,.2f}"

# ==============================
# USER QUERY
# ==============================
user_query = f"""
Budget : {budget}
Category : {category}
Brand : {brand if brand else "Any"}
Minimum Rating : {rating}

Requirements :
{requirements}
"""
# ===========================================
# TOOL 1 : SEARCH LATEST PRODUCTS USING TAVILY
# ===========================================
def search_products(query: str) -> dict:
    """Search latest products, reviews and buying guides."""
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        return client.search(query=query, search_depth="advanced", max_results=5)
    except Exception as e:
        st.warning(f"Tavily search failed: {e}")
        return {"results": []}
# ===========================================
# TOOL 2 : GET PRODUCTS
# ===========================================
def get_products(category: str) -> list:
    """Fetch products from FakeStore API."""
    try:
        url = f"https://fakestoreapi.com/products/category/{category}"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException as e:
        st.warning(f"Could not fetch products: {e}")
    return []
# ===========================================
# TOOL 2B : LIVE BUY LINK + PHOTO + REVIEWS (AMAZON / FLIPKART)
# ===========================================
@st.cache_data(ttl=1800, show_spinner=False)
def get_live_buy_info(product_title: str) -> dict:
    """
    Use Tavily to find a real, live Amazon/Flipkart product page for the
    given title, along with a product photo and short review snippets.
    Returns: {"buy_url": str, "buy_site": str, "image": str, "reviews": [str, ...]}
    """
    info = {"buy_url": "", "buy_site": "", "image": "", "reviews": []}
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        query = f"{product_title} buy price reviews site:amazon.in OR site:flipkart.com"
        res = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_images=True,
        )

        # Pick the first Amazon/Flipkart result as the buy link
        for r in res.get("results", []):
            url = r.get("url", "")
            if "amazon." in url or "flipkart.com" in url:
                info["buy_url"] = url
                info["buy_site"] = "Amazon" if "amazon." in url else "Flipkart"
                if r.get("content"):
                    info["reviews"].append(r["content"][:300])
            elif r.get("content"):
                info["reviews"].append(r["content"][:300])

        # Fallback: first result of any kind if no marketplace link found
        if not info["buy_url"] and res.get("results"):
            info["buy_url"] = res["results"][0].get("url", "")
            info["buy_site"] = "Web"

        images = res.get("images", [])
        if images:
            info["image"] = images[0] if isinstance(images[0], str) else images[0].get("url", "")

    except Exception as e:
        st.warning(f"Live buy-link search failed for '{product_title}': {e}")

    return info

# ===========================================
# TOOL 3 : BUDGET / RATING / BRAND FILTER
# ===========================================
def filter_products(products, budget, min_rating=1, brand=""):
    """Filter products by budget, minimum rating and (optional) brand keyword."""
    result = []
    brand_lower = brand.strip().lower()
    
    for product in products:
        price = float(product.get("price", 0))
        product_rating = float(product.get("rating", {}).get("rate", 0))
        title = product.get("title", "").lower()

        if price > budget:
            continue
        if product_rating < min_rating:
            continue
        if brand_lower and brand_lower not in title:
            continue

        result.append(product)

    return result
# ===========================================
# TOOL 4 : PRODUCT COMPARISON
# ===========================================
def compare_products(products):
    """Compare products using Gemini."""
    if not products:
        return "No products available to compare."
    prompt = f"""
You are an AI Shopping Assistant.
Compare the following products.
Return output as a Markdown table with columns:
Price | Features | Pros | Cons | Rating | Best Choice
Products:
{products}
"""
    response = model.invoke(prompt)
    return response.content
# ===========================================
# TOOL 5 : PRODUCT RECOMMENDATION
# ===========================================
def recommend_products(products, requirements):
    """Recommend best products based on user requirements."""
    if not products:
        return "No products available to recommend."
    prompt = f"""
You are an AI Shopping Recommendation Assistant.

Requirements:
{requirements}
Products:
{products}
Recommend Top 5 Products.
For every product provide (in Markdown):
- Product Name
- Price
- Why Recommended
- Pros
- Cons
- Buying Score out of 10
"""
    response = model.invoke(prompt)
    return response.content
# ===========================================
# TOOL 6 : ALTERNATIVE PRODUCTS
# ===========================================
def alternative_products(products):
    """Suggest alternative products."""
    if not products:
        return "No products available to suggest alternatives for."
    prompt = f"""
Suggest affordable alternatives (in Markdown) for the following products.
Products:
{products}
"""
    response = model.invoke(prompt)
    return response.content
# ===========================================
# HELPER: clean model output before rendering
# ===========================================
def clean_llm_output(text) -> str:
    """Strip ```html / ``` code fences the model sometimes wraps output in.
    Defensive against non-string input: some LangChain/Gemini responses
    return `content` as None or as a list of content-part dicts instead
    of a plain string.
    """
    if text is None:
        return ""
    if isinstance(text, list):
        parts = []
        for part in text:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        text = "\n".join(p for p in parts if p)
        
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = re.sub(r"^```(?:html|markdown)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    
    text = "\n".join(line.lstrip() for line in text.splitlines())

    return text
# ===========================================
# CREATE LEADER AGENT
# ===========================================
leader_agent = create_agent(
    model=model,
    tools=[search_products, get_products]
)
# ===========================================
# MAIN SHOPPING AGENT
# ===========================================
def shopping_assistant(agent, query):
    """Leader Agent responsible for generating the final shopping recommendation."""
    prompt = f"""
You are Smart Shopping Recommendation Assistant.
Your responsibilities are:
1. Understand customer requirements.
2. Understand budget.
3. Recommend only products within budget.
4. Compare products.
5. Suggest alternatives.
6. Explain why product is best.
7. Mention Pros.
8. Mention Cons.
9. Give Buying Verdict.
10. Show latest product information if required.
Return output only in clean HTML (no markdown, no code fences).
User Query:
{query}
"""

    response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )

    try:
        content = response["messages"][-1].content
    except (KeyError, IndexError, AttributeError, TypeError) as e:
        st.warning(f"Unexpected agent response shape ({e}); showing raw output.")
        content = str(response)

    return clean_llm_output(content)
# ===========================================
# LOAD PRODUCTS
# ===========================================
products = get_products(category)
filtered_products = filter_products(products, budget, rating, brand)
# ===========================================
# BUTTON
# ===========================================
if st.button("🛒 Recommend Products", use_container_width=True):
    with st.spinner("Finding Best Products..."):

        latest = search_products(f"{category} best products under {budget}")
        latest_results = latest.get("results", [])

        try:
            html = shopping_assistant(leader_agent, user_query)
        except Exception as e:
            st.error(f"Gemini request failed: {e}")
            st.stop()

        st.success("Recommendation Generated Successfully")
        st.markdown("## AI Shopping Recommendation")

        cleaned_html = clean_llm_output(html)
        components.html(cleaned_html, height=900, scrolling=True)

        st.divider()
        st.subheader("Products From FakeStore API")
        st.caption(f"Exchange rate used: 1 USD = ₹{USD_TO_INR:,.2f}")

        if len(filtered_products) == 0:
            st.warning("No products found matching your budget / rating / brand filters.")
        else:
            cols = st.columns(2)
            for index, product in enumerate(filtered_products):
                with cols[index % 2]:
                    live_info = get_live_buy_info(product["title"])
                    display_image = live_info.get("image") or product["image"]

                    st.image(display_image, width=180)
                    st.markdown(f"### {product['title']}")
                    st.write(product["description"][:180])
                    st.metric("Price", format_price(product["price"]))
                    st.write("⭐", product["rating"]["rate"], f"({product['rating']['count']} ratings)")

                    if live_info.get("buy_url"):
                        st.link_button(
                            f"🛒 Buy on {live_info.get('buy_site', 'Store')}",
                            live_info["buy_url"],
                            use_container_width=True,
                        )
                    else:
                        st.caption("No live buy link found.")

                    if live_info.get("reviews"):
                        with st.expander("📝 User Reviews (from web)"):
                            for rev in live_info["reviews"][:3]:
                                st.write("•", rev)

        st.divider()
        st.subheader("Latest Shopping Results")

        if len(latest_results) == 0:
            st.info("No latest search results found.")
        else:
            for item in latest_results:
                title = item.get("title", "No Title")
                url = item.get("url", "")
                content = item.get("content", "")

                with st.expander(title):
                    st.write(content)
                    if url:
                        st.link_button("Open Website", url)

        st.divider()
        st.subheader("Product Comparison")
        comparison = compare_products(filtered_products)
        st.markdown(clean_llm_output(comparison))

        st.divider()
        st.subheader("Alternative Products")
        alt = alternative_products(filtered_products)
        st.markdown(clean_llm_output(alt))

        st.divider()
        st.success("Shopping Recommendation Completed")
