import json
import random
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class RLAgent:
    def __init__(self, n_articles, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, min_epsilon=0.01, decay_rate=0.005, q_table_file=None):
        self.q_table_file = q_table_file  # Ensure this is set properly
        if self.q_table_file and os.path.exists(self.q_table_file):
            self.load_q_table()
            if len(self.q_table) != n_articles:
                print(f"Q-table size mismatch: resetting to {n_articles} entries.")
                self.q_table = [0] * n_articles  # Reset the q_table to match the number of articles
        else:
            self.q_table = [0] * n_articles  # Initialize the q_table with the correct size

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.total_reward = 0  # Initialize total reward

        self.analyzer = SentimentIntensityAnalyzer()  # Initialize VADER analyzer

    def choose_action(self):
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, len(self.q_table) - 1)  # Explore
        else:
            action = self.q_table.index(max(self.q_table))  # Exploit
        self._decay_epsilon()
        return action

    def update(self, article_index, reward):
        current_q = self.q_table[article_index]
        max_future_q = max(self.q_table)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[article_index] = new_q
        self.total_reward += reward  # Update total reward
        self.save_q_table()  # Save after update

    def _decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= (1 - self.decay_rate)

    def save_q_table(self):
        if self.q_table_file:
            with open(self.q_table_file, 'w') as file:
                json.dump(self.q_table, file)

    def load_q_table(self):
        if self.q_table_file and os.path.exists(self.q_table_file):
            with open(self.q_table_file, 'r') as file:
                self.q_table = json.load(file)
        else:
            print("Q-table file not found, initializing with zeros.")

    def analyze_sentiment(self, feedback_text):
        # Perform sentiment analysis using VADER
        sentiment_score = self.analyzer.polarity_scores(feedback_text)['compound']
        return "positive" if sentiment_score > 0 else "negative"

def load_news_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data
