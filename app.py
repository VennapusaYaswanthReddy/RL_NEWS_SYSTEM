from flask import Flask, render_template, request, redirect, url_for
from rl_model import RLAgent, load_news_data
import random

app = Flask(__name__)

# Load news articles
news_articles = load_news_data('news_data.json')
print(f"Loaded {len(news_articles)} news articles!")

# Initialize RL Agent with 100 articles
agent = RLAgent(len(news_articles), q_table_file='q_table.json')
print(f"Q-table size: {len(agent.q_table)}")

# Track the current article index and seen articles
current_index = 0
seen_articles = set()

@app.route('/')
def home():
    global current_index

    if not news_articles:
        return "No news articles available."

    # Safety check: reset current_index if invalid
    if current_index < 0 or current_index >= len(news_articles):
        current_index = random.randint(0, len(news_articles) - 1)

    article = news_articles[current_index]
    return render_template('index.html', article=article, total_reward=agent.total_reward, index=current_index)

@app.route('/feedback', methods=['POST'])
def feedback():
    global current_index, seen_articles
    action = request.form.get('action')  # Use .get() to avoid KeyError

    # Get feedback text (if any)
    feedback_text = request.form.get('feedback', '').strip()  # Default to empty string if no feedback is provided

    # Check if feedback is missing
    if not feedback_text:
        # If no feedback, use a neutral sentiment (optional handling)
        sentiment = "neutral"
    else:
        # Analyze sentiment of the feedback
        sentiment = agent.analyze_sentiment(feedback_text)
    
    # Determine reward based on sentiment
    if sentiment == "positive":
        reward = 10
    elif sentiment == "negative":
        reward = -10
    else:
        reward = 0

    # Update only if current_index is valid
    if 0 <= current_index < len(news_articles):
        agent.update(current_index, reward)
        agent.save_q_table()
    else:
        print(f"Warning: Skipping invalid current_index {current_index}")

    # Mark current article as seen
    seen_articles.add(current_index)

    # Pick a NEW unseen article
    unseen_articles = list(set(range(len(news_articles))) - seen_articles)
    if unseen_articles:
        current_index = random.choice(unseen_articles)
    else:
        seen_articles.clear()
        current_index = random.randint(0, len(news_articles) - 1)

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
