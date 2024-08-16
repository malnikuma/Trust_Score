import pandas as pd
import openai
import time

# Configure the OpenAI API key
openai.api_key = "apikey"

categories = [
    "Very Negative",
    "Clearly Negative",
    "Quite Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Quite Positive",
    "Clearly Positive",
    "Very Positive"
]

sentiment_mapping  = {
    "Very Negative": 0.0,
    "Clearly Negative": 0.25,
    "Quite Negative": 0.35,
    "Negative": 0.45,
    "Neutral": 0.5,
    "Positive": 0.55,
    "Quite Positive": 0.65,
    "Clearly Positive": 0.75,
    "Very Positive": 1.0
}

def analyze_sentiment(text):
    messages = [
        {"role": "system", "content": "You are a sentiment analysis model."},
        {"role": "user", "content": f"Analyze the sentiment of the following feedback and classify it into one of these categories. Return only the category without any additional text:\n"
                                    "Very Negative\n"
                                    "Clearly Negative\n"
                                    "Quite Negative\n"
                                    "Negative\n"
                                    "Neutral\n"
                                    "Positive\n"
                                    "Quite Positive\n"
                                    "Clearly Positive\n"
                                    "Very Positive\n\n"
                                    f"Feedback: {text}"}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=10,
        n=1,
        temperature=0.7,
    )
    sentiment = response.choices[0].message.content.strip()
    time.sleep(0.5)
    sentiment_score = sentiment_mapping.get(sentiment, 0.5)
    print(f'{sentiment} : {sentiment_score}')
    return sentiment_score
   
# Load the CSV file
file_path = 'transactions1.csv'
df = pd.read_csv(file_path)

# Assign the weights to the transaction types
weights = {1: 0.2, 2: 0.3, 3: 0.5}

# Add a column for weighted success (success * weight)
df['weighted_success'] = df['transaction_type'].map(weights) * df['success']

# Group by farmer_id and calculate the total weighted success and total possible weight
grouped = df.groupby('farmer_id').agg({'weighted_success': 'sum'})
grouped['total_possible_weight'] = df['transaction_type'].map(weights).groupby(df['farmer_id']).sum()

# Calculate the success rate for each farmer (weighted success / total possible weight)
grouped['success_rate'] = grouped['weighted_success'] / grouped['total_possible_weight']

# Merge the success rate back into the original dataframe
df = df.merge(grouped[['success_rate']], on='farmer_id')

# Calculate sentiment score for each feedback
df['sentiment_score'] = df['feedback'].apply(analyze_sentiment)

# Normalize the rating to a score between 0 and 1
df['normalized_rating'] = df['rating'] / 5

# Calculate the trust score for each row
df['trust_score'] = (0.4 * df['sentiment_score']) + (0.3 * df['normalized_rating']) + (0.3 * df['success_rate'])

# Aggregate trust score by farmer
final_result = df.groupby('farmer_id').agg({
    'sentiment_score': 'mean',
    'normalized_rating': 'mean',
    'success_rate': 'mean',
    'trust_score': 'mean'
}).reset_index()

# Export the final result to a CSV file
output_file_path = 'openai_trust_scores.csv'
final_result.to_csv(output_file_path, index=False)

# Display the final result
print(final_result)