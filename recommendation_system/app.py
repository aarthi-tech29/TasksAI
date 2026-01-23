# ====================================
# IMPORT REQUIRED LIBRARIES
# ====================================
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# pandas → table data
# numpy → numerical matrix
# cosine_similarity → find similar users

# ====================================
# CREATE SAMPLE DATASET
# ====================================
# Example: OTT platform (movies)
data = {
    "User": ["A", "A", "A", "B", "B", "C", "C", "D"],
    "Movie": ["M1", "M2", "M3", "M1", "M3", "M2", "M4", "M4"],
    "Rating": [5, 4, 3, 5, 4, 2, 5, 4]
}

df = pd.DataFrame(data)
print(df)

# User A rated M1, M2, M3
# User C loves M4
# User D watched only M4

# ====================================
# CREATE USER-MOVIE MATRIX
# ====================================
user_item_matrix = df.pivot_table(
    index="User",
    columns="Movie",
    values="Rating"
).fillna(0)

print("\nUser-Item Matrix:")
print(user_item_matrix)

# Rows = Users
# Columns = Movies
# Values = Ratings
# 0 = Not watched
# ====================================
# Calculate User Similarity
# ===================================
# Using Cosine Similarity

similarity_matrix = cosine_similarity(user_item_matrix)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

print("\nUser Similarity Matrix:")
print(similarity_df)

# Value close to 1 → very similar users
# Value close to 0 → different tastes

# ====================================
# Recommendation Logic
# ====================================

# Items liked by similar users
# Items the target user has not rated

def recommend(user, matrix, similarity_df, top_n=2):
    # Get similarity scores of the user
    similar_users = similarity_df[user].sort_values(ascending=False)
    
    # Remove the user itself
    similar_users = similar_users.drop(user)
    
    recommendations = {}

    for sim_user in similar_users.index: # For each similar user
        user_ratings = matrix.loc[sim_user] # Get their ratings of similar user
        for item, rating in user_ratings.items(): # For each item they rated
            if matrix.loc[user][item] == 0 and rating > 0: # If target user hasn't rated,watched this movie and Similar user HAS watched it
                recommendations[item] = recommendations.get(item, 0) + rating # Score it to recommendations

    # Sort recommendations
    recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True) # sort and best recommendation
    
    return recommendations[:top_n] 

# Target User -> Find Similar Users-> Check What They Watched->Exclude Already Seen Items->Score & Rank->Recommend Top Items

# ======================================
# Run Recommendation
# =====================================
user_name = "A"
recommended_items = recommend(user_name, user_item_matrix, similarity_df)

print(f"Recommendations for User {user_name}:")
for item, score in recommended_items:
    print(f"- {item}")

# User A never watched M4, but similar users liked it → recommended
