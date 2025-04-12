import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Load data
match_df = pd.read_csv("cleaned_match_info.csv", low_memory=False)
delivery_df = pd.read_csv("cleaned_deliveries.csv", low_memory=False)

# Merge datasets
df = delivery_df.merge(match_df, on='match_id')

# Sample for faster testing
df = df.sample(frac=0.3, random_state=42)

# Fill missing values
df.fillna(0, inplace=True)

# Optimize memory usage
df['batter'] = df['batter'].astype('category')
df['bowler'] = df['bowler'].astype('category')

# KMeans clustering to simulate delivery types
features = df[['over', 'ball_number', 'pressure_index', 'bowler_economy']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42)
df['delivery_type'] = kmeans.fit_predict(X_scaled)

# Batter performance aggregation
batter_stats = df.groupby(['batter', 'delivery_type'], observed=True).agg({
    'runs_batter': ['sum', 'mean'],
    'ball_number': 'count',
    'batter_strike_rate': 'mean'
}).reset_index()

# Flatten column names
batter_stats.columns = ['batter', 'delivery_type', 'total_runs', 'avg_runs', 'balls_faced', 'avg_strike_rate']

# Bowler delivery type trends
bowler_trends = df.groupby(['bowler', 'delivery_type'], observed=True).size().unstack(fill_value=0).reset_index()

# Average bowler economy
bowler_economy = df.groupby('bowler', observed=True)['bowler_economy'].mean().reset_index()

# Export CSV outputs
batter_stats.to_csv("output_batter_patterns.csv", index=False)
bowler_trends.to_csv("output_bowler_trends.csv", index=False)
bowler_economy.to_csv("output_bowler_economy.csv", index=False)

print("✅ Analysis complete. CSVs saved.")

# Heatmap of avg strike rate for top 20 batters
try:
    top_batters = (
        batter_stats.groupby('batter', observed=True)['total_runs']
        .sum()
        .nlargest(20)
        .index
    )
    top_batter_stats = batter_stats[batter_stats['batter'].isin(top_batters)].copy()

    # Clip extreme strike rates
    top_batter_stats['avg_strike_rate'] = top_batter_stats['avg_strike_rate'].clip(upper=250)

    pivot = top_batter_stats.pivot(index="batter", columns="delivery_type", values="avg_strike_rate")

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.3, linecolor='gray')
    plt.title("Avg Strike Rate by Delivery Type (Top 20 Batters)", fontsize=14)
    plt.xlabel("Delivery Type")
    plt.ylabel("Batter")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("batter_strike_rate_heatmap_top20.png")
    plt.close()
except Exception as e:
    print("⚠️ Strike rate heatmap generation failed:", e)

# Horizontal bar chart for bowler delivery types (Top 20)
try:
    # Add total column and get top 20
    bowler_trends['total'] = bowler_trends[[0, 1, 2, 3]].sum(axis=1)
    top_bowler_trends = bowler_trends.sort_values(by='total', ascending=False).head(20)

    # Melt for plotting
    bowler_long = top_bowler_trends.melt(
        id_vars=['bowler'],
        value_vars=[0, 1, 2, 3],
        var_name='delivery_type',
        value_name='count'
    )

    # Sort y-axis by total deliveries
    bowler_order = (
        bowler_long.groupby('bowler', observed=True)['count']
        .sum()
        .sort_values(ascending=True)
        .index
    )

    # Plot
    plt.figure(figsize=(12, 10))
    sns.barplot(
        data=bowler_long,
        y='bowler',
        x='count',
        hue='delivery_type',
        order=bowler_order,
        palette='magma'
    )

    plt.title("Bowler Delivery Type Distribution (Top 20 Bowlers)", fontsize=14)
    plt.xlabel("Delivery Count", fontsize=12)
    plt.ylabel("Bowler", fontsize=12)
    plt.legend(title="Delivery Type")
    plt.tight_layout()
    plt.savefig("bowler_delivery_distribution_top20.png")
    plt.close()

except Exception as e:
    print("⚠️ Horizontal bar chart generation failed:", e)