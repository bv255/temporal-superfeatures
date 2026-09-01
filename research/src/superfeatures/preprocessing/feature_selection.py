"""
Ported from `research/PreProcessing_test.ipynb` cell 4 — see `docs/RESTRUCTURING_TODO.md`
and the port plan in `docs/RESEARCH_STRUCTURE.md`. Extracted mechanically from the notebook
cell source (not retyped) to avoid transcription drift; the notebook remains the frozen parity
reference. Only the import block was touched: names the cell relied on getting from the shared
notebook kernel namespace (star-imports in cell 0, e.g. `from pyspark.sql.types import *`) are
now imported explicitly here, since a standalone module has no such shared kernel state.
"""

"""This is the class for Experiment 1: Feature Selection in which 
statistical measures are computed and applied to the dataset processed
by the Utils class above
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import Tuple, List

class FeatureSelection:
    def __init__(self, df: DataFrame, target_column: str, num_partitions: int = 24):
        self.df = df.repartition(num_partitions)
        self.target = target_column
        self.non_feature_columns = ['date', 'fsym', 'factset_sector_desc', 'target_date', 'return_year', 'return_month', 'monthly_return'] # monthly_return is the target
        self.features = [col for col in df.columns if col not in self.non_feature_columns]
        self.features_to_remove = []  # Features to remove for the genetic algorithm
        self._row_count_cache = None  # self.df's row count is never changed by the methods below (they only ever
        # select/drop columns, never filter rows), so it is computed once via self._rows() and reused everywhere
        # instead of re-triggering a full Spark count() at every dimensionality-reduction step.

        print(f"FeatureSelection init: rows={self._rows()}, features={len(self.features)}")

    def _rows(self) -> int:
        """Cached row count of self.df (see note in __init__ on why this is safe to cache)."""
        if self._row_count_cache is None:
            self._row_count_cache = self.df.count()
        return self._row_count_cache

    def pearson_correlation(self, bottom_pct: float = 0.1, plot: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Computing and plotting the feature Pearson correlation with respect to the 'monthly_return'
        column, which is the one-month forward return. Also, identify features with the lowest
        10% correlation values for removal.

        Parameters:
        - bottom_pct: fraction of features (ranked by absolute correlation, ascending) flagged
          as removal candidates. Default 0.1 (bottom 10%). Nothing is actually dropped from
          self.df here - candidates are only applied once filter_features() is called.

        Returns:
        - pd.DataFrame: DataFrame of Pearson correlations for plotting.
        - List[str]: List of features with the lowest 10% correlation values to be removed.
        """
        print("pearson_correlation: starting...")
        rows_before, features_before = self._rows(), len(self.features)
        print(f"pearson_correlation: input -> rows={rows_before}, features={features_before}")
        print(f"pearson_correlation params: bottom_pct={bottom_pct} (fraction of features flagged as removal candidates)")

        correlations = self.df.select(
            *[F.corr(F.col(feature), F.col(self.target)).alias(feature) for feature in self.features]
        ).collect()[0].asDict()
        
        print("Converting correlation set to Pandas for plotting...")

        # Convert to Pandas for easier manipulation and plotting
        pearson_corr_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Pearson Correlation'])
        
        # Identify the bottom `bottom_pct` features based on absolute correlation values
        threshold_index = int(bottom_pct * len(pearson_corr_df))
        bottom_10_percent_features = pearson_corr_df['Feature'].iloc[pearson_corr_df['Pearson Correlation'].abs().sort_values().index[:threshold_index]].tolist()

        # Sort by absolute value of the correlations and take the top 50
        pearson_corr_df = pearson_corr_df.reindex(pearson_corr_df['Pearson Correlation'].abs().sort_values(ascending=False).index).head(50)

        if plot:
            print("Plotting:")

            plt.figure(figsize=(12, 8))
            sns.barplot(x='Feature', y='Pearson Correlation', data=pearson_corr_df)
            plt.xticks(rotation=90)
            plt.title('Top 50 Pearson Correlations of features with Monthly Returns')

            # Save the plot as a PNG file
            plt.savefig("top_50_pearson_correlations.png", dpi=300, bbox_inches='tight')

            plt.show()

        print(f"pearson_correlation: {len(bottom_10_percent_features)} of {features_before} features flagged for removal "
              f"(rows unchanged at {rows_before}, features unchanged at {features_before} until filter_features() runs)")
        print("pearson_correlation: done.")
        return pearson_corr_df, bottom_10_percent_features

    def spearman_correlation(self, sample_fraction: float = 0.1, bottom_pct: float = 0.1, plot: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Computing and plotting the feature Spearman correlation with respect to the 'monthly_return'
        column, which is the one-month forward return.
        Also, identify features with the lowest
        10% correlation values for removal.

        Parameters:
        - sample_fraction: fraction of companies (fsym) sampled before computing the Spearman
          correlation matrix, for efficiency. Default 0.1 (10% of companies).
        - bottom_pct: fraction of features (ranked by absolute correlation, ascending) flagged
          as removal candidates. Default 0.1 (bottom 10%).

        Returns:
        - pd.DataFrame: DataFrame of Spearman correlations for plotting.
        - List[str]: List of features with the lowest 10% correlation values to be removed.
    
        """
        def sample_companies(df, fraction=0.1):
            unique_companies = df.select('fsym').distinct().collect()
            unique_companies = [row['fsym'] for row in unique_companies]
            sample_size = int(len(unique_companies) * fraction)
            sampled_companies = random.sample(unique_companies, sample_size)
            filtered_df = df.filter(F.col('fsym').isin(sampled_companies))
            return filtered_df

        print("spearman_correlation: starting...")
        rows_before, features_before = self._rows(), len(self.features)
        print(f"spearman_correlation: input -> rows={rows_before}, features={features_before}")
        print(f"spearman_correlation params: sample_fraction={sample_fraction}, bottom_pct={bottom_pct}")

        print(f"Sampling {sample_fraction:.0%} of the companies for efficiency")
        sampled_df = sample_companies(self.df, fraction=sample_fraction).cache()
        rows_sampled = sampled_df.count()
        print(f"spearman_correlation: rows before company sampling={rows_before}, rows after sampling={rows_sampled} "
              f"({rows_sampled / rows_before:.1%} of rows kept; this sampled frame is local to this method and does not affect self.df)")
        
        # Handle null values by filling them with 0
        sampled_df = sampled_df.fillna(0, subset=self.features + [self.target])

        # Assemble the features into a vector
        assembler = VectorAssembler(inputCols=self.features + [self.target], outputCol="features")
        sampled_df = assembler.transform(sampled_df).select("features")

        # Compute the correlation matrix using Spearman method
        corr_matrix = Correlation.corr(sampled_df, "features", method="spearman").head()[0].toArray()
        
        # Convert to DataFrame for easier handling
        corr_df = pd.DataFrame(corr_matrix, columns=self.features + [self.target], index=self.features + [self.target])
        
        # Extract correlations with the target
        spearman_corr_df = corr_df[self.target].drop(self.target).reset_index()
        spearman_corr_df.columns = ['Feature', 'Spearman Correlation']
        
        # Identify the bottom `bottom_pct` features based on absolute correlation values
        threshold_index = int(bottom_pct * len(spearman_corr_df))
        bottom_10_percent_features = spearman_corr_df['Feature'].iloc[spearman_corr_df['Spearman Correlation'].abs().sort_values().index[:threshold_index]].tolist()

        # Sort by absolute value of the correlations and take the top 50
        spearman_corr_df = spearman_corr_df.reindex(spearman_corr_df['Spearman Correlation'].abs().sort_values(ascending=False).index).head(50)

        if plot:
            print("Plotting:")

            plt.figure(figsize=(12, 8))
            sns.barplot(x='Feature', y='Spearman Correlation', data=spearman_corr_df)
            plt.xticks(rotation=90)
            plt.title('Top 50 Spearman Correlations of features with Monthly Returns')
            plt.savefig("top_50_spearman_correlations.png", dpi=300, bbox_inches='tight')
            plt.show()

        print(f"spearman_correlation: {len(bottom_10_percent_features)} of {features_before} features flagged for removal "
              f"(self.df rows/features unchanged until filter_features() runs)")
        print("spearman_correlation: done.")
        return spearman_corr_df, bottom_10_percent_features
    
    def filter_features(self, pearson_remove: List[str], spearman_remove: List[str]) -> DataFrame:
        """
        Filter features from the DataFrame based on Pearson and Spearman correlation thresholds.
        This is where the removal candidates identified by pearson_correlation() and
        spearman_correlation() are actually applied to self.df.

        Args:
        - pearson_remove: List of features flagged by pearson_correlation().
        - spearman_remove: List of features flagged by spearman_correlation().

        Returns:
        - DataFrame: DataFrame with specified features removed.
        """

        print("filter_features: starting...")
        rows_before, features_before = self._rows(), len(self.features)
        print(f"filter_features: before -> rows={rows_before}, features={features_before}")

        # Combine and deduplicate features to remove
        pearson_set, spearman_set = set(pearson_remove), set(spearman_remove)
        features_to_remove = pearson_set | spearman_set
        pearson_only = pearson_set - spearman_set
        spearman_only = spearman_set - pearson_set
        flagged_by_both = pearson_set & spearman_set

        # Add removed features to the list
        self.features_to_remove.extend(features_to_remove)

        print(f"filter_features params: Pearson-flagged={len(pearson_remove)}, Spearman-flagged={len(spearman_remove)}, "
              f"unique union to remove={len(features_to_remove)}")
        print(f"filter_features breakdown: removed only-by-Pearson={len(pearson_only)}, "
              f"only-by-Spearman={len(spearman_only)}, by-both={len(flagged_by_both)}")

        # Remove features from DataFrame
        remaining_features = [col for col in self.df.columns if col not in features_to_remove or col in self.non_feature_columns]
        self.df = self.df.select(*remaining_features)
        
        # Update self.features
        self.features = [feature for feature in self.features if feature not in features_to_remove]

        features_after = len(self.features)
        # rows_after == rows_before: this step only ever selects a subset of columns, never filters rows
        print(f"filter_features: after -> rows={rows_before}, features={features_after}")
        print(f"filter_features: done. Removed {features_before - features_after} features, rows unchanged.")
        return self.df

    def mutual_information(self, top_n: int = 50, sample_fraction: float = 0.1, plot: bool = True) -> pd.DataFrame:
        """
        Computing and plotting the feature Mutual Information with respect to the 'monthly_return'
        column, which is the one-month forward return. This step only scores features - it does
        not remove any (removal happens later, in remove_high_corr_features_based_on_mi()).

        Parameters:
        - top_n: number of top-scoring features to plot. Does not affect the returned
          DataFrame, which always contains scores for every feature.
        - sample_fraction: fraction of companies (fsym) sampled before converting to Pandas for
          sklearn's mutual_info_regression - same idea as spearman_correlation()'s sampling,
          avoids pulling the full dataset to the driver. Default 0.1 (10% of companies).
        """
        print("mutual_information: starting...")
        features_before = len(self.features)
        print(f"mutual_information: input -> rows={self._rows()}, features={features_before}")
        print(f"mutual_information params: top_n={top_n}, sample_fraction={sample_fraction} "
              f"(features plotted only, none removed at this step)")

        # Sample companies before converting to Pandas, rather than pulling the full dataset to
        # the driver - mirrors the sampling already done in spearman_correlation().
        unique_companies = [row['fsym'] for row in self.df.select('fsym').distinct().collect()]
        sample_size = min(len(unique_companies), max(1, int(len(unique_companies) * sample_fraction)))
        sampled_companies = random.sample(unique_companies, sample_size)
        sampled_df = self.df.filter(F.col('fsym').isin(sampled_companies))

        # Select features and target columns and convert to Pandas
        pandas_df = sampled_df.select(*self.features, self.target).toPandas()
        
        X = pandas_df[self.features].copy()
        y = pandas_df[self.target].copy()
        
        print(f"Switched to Pandas! rows used={len(pandas_df)} ({len(sampled_companies)}/{len(unique_companies)} companies sampled), "
              f"features scored={len(self.features)}")

        # Fill NA values
        X.fillna(0, inplace=True)
        y.fillna(0, inplace=True)

        # Compute mutual information
        mutual_info = mutual_info_regression(X, y)
        print("Computed mutual information!")
        mutual_info_df = pd.DataFrame(mutual_info, index=self.features, columns=['Mutual Information'])

        # Sort by mutual information score
        mutual_info_df = mutual_info_df.sort_values(by='Mutual Information', ascending=False)
        
        if plot:
            # Plotting the top N features
            top_mutual_info_df = mutual_info_df.head(top_n)

            plt.figure(figsize=(12, 8))
            sns.barplot(x=top_mutual_info_df.index, y='Mutual Information', data=top_mutual_info_df)
            plt.xticks(rotation=90)
            plt.xlabel("Feature")
            plt.title(f'Top {top_n} Mutual Information Scores with Monthly Returns')
            plt.savefig(f"top_{top_n}_mutual_information.png", dpi=300, bbox_inches='tight')
            plt.show()

        print(f"mutual_information: done. features unchanged at {features_before} (no removal at this step).")
        return mutual_info_df
    
    def remove_high_corr_features_based_on_mi(self, mutual_info_df: pd.DataFrame, corr_threshold: float = 0.8, sample_fraction: float = 0.1, plot: bool = True) -> DataFrame:
        """
        Remove highly correlated features, keeping the one with higher mutual information.

        Parameters:
        - mutual_info_df (pd.DataFrame): DataFrame containing features and their mutual information scores.
        - corr_threshold (float): Correlation threshold above which features are considered highly correlated.
          Default 0.8. Of each correlated pair, the feature with the lower mutual information score is dropped.
        - sample_fraction (float): fraction of companies (fsym) sampled before converting to Pandas -
          the same sample is reused for both the initial and post-removal correlation matrices below,
          avoiding two full-dataset pulls to the driver. Default 0.1 (10% of companies).

        Returns:
        - DataFrame: DataFrame with reduced features.
        """
        print("Starting the removal of highly correlated features...")
        rows_before = self._rows()
        features_before = len(self.features)
        print(f"remove_high_corr_features_based_on_mi: input -> rows={rows_before}, features={features_before}")
        print(f"remove_high_corr_features_based_on_mi params: corr_threshold={corr_threshold}, sample_fraction={sample_fraction}")

        # Sample companies before converting to Pandas, rather than pulling the full dataset to
        # the driver (this DataFrame is reused below for the post-removal correlation matrix too,
        # so it's cached rather than resampled a second time) - mirrors spearman_correlation().
        unique_companies = [row['fsym'] for row in self.df.select('fsym').distinct().collect()]
        sample_size = min(len(unique_companies), max(1, int(len(unique_companies) * sample_fraction)))
        sampled_companies = random.sample(unique_companies, sample_size)
        sampled_df = self.df.filter(F.col('fsym').isin(sampled_companies)).cache()

        # Select features and target columns and convert to Pandas
        print("Converting Spark DataFrame to Pandas DataFrame...")
        pandas_df = sampled_df.select(*self.features).toPandas()
        print(f"Conversion complete! rows converted={len(pandas_df)} ({len(sampled_companies)}/{len(unique_companies)} companies sampled)")

        # Compute the correlation matrix
        print("Computing the correlation matrix...")
        corr_matrix = pandas_df.corr()
        print("Correlation matrix computation complete!")

        if plot:
            # Save the initial correlation matrix as an image
            initial_corr_matrix_image_path = "initial_correlation_matrix.png"
            print(f"Saving the initial correlation matrix as an image to {initial_corr_matrix_image_path}...")
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title("Initial Correlation Matrix")
            plt.savefig(initial_corr_matrix_image_path)
            plt.close()
            print("Initial correlation matrix image saved!")

        # Identify pairs of features with correlation higher than the threshold
        print(f"Identifying pairs of features with correlation higher than {corr_threshold}...")
        
        correlated_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if np.abs(corr_matrix.iloc[i, j]) > corr_threshold:
                    colname_i = corr_matrix.columns[i]
                    colname_j = corr_matrix.columns[j]
                    correlated_pairs.append((colname_i, colname_j))
                    
        print(f"Found {len(correlated_pairs)} pairs of highly correlated features (threshold={corr_threshold}).")

        # Determine which features to drop based on mutual information
        print("Determining which features to drop based on mutual information...")
        features_to_drop = set()
        
        for pair in correlated_pairs:
            feature1, feature2 = pair
            importance1 = mutual_info_df.loc[feature1, "Mutual Information"]
            importance2 = mutual_info_df.loc[feature2, "Mutual Information"]
            if importance1 < importance2:
                features_to_drop.add(feature1)
            else:
                features_to_drop.add(feature2)
        
        print(f"Number of features to drop: {len(features_to_drop)}")

        # Remove features from the DataFrame
        print("Removing the selected features from the DataFrame...")
        
        remaining_features = [col for col in self.features if col not in features_to_drop]
        remaining_features += self.non_feature_columns
        self.df = self.df.select(*remaining_features)
        
        # Add removed features to the list
        self.features_to_remove.extend(features_to_drop)

        # Update self.features so later calls (and any before/after reporting) reflect the columns
        # actually left in self.df - this mirrors what filter_features() already does, and matters
        # because calling this method again on a stale self.features would try to select columns
        # that were already dropped.
        self.features = [feature for feature in self.features if feature not in features_to_drop]

        features_after = len(self.features)
        # rows_after == rows_before: this step only ever selects a subset of columns, never filters rows
        print(f"remove_high_corr_features_based_on_mi: after -> rows={rows_before}, features={features_after}")
        print(f"remove_high_corr_features_based_on_mi: done. Removed {features_before - features_after} features, rows unchanged.")

        if plot:
            # Recompute the correlation matrix with the remaining features
            # NOTE: uses self.features (numeric only), not remaining_features - the latter also
            # includes non_feature_columns (date/fsym/factset_sector_desc/...) which are kept in
            # self.df intentionally but aren't valid input to pandas .corr() (non-numeric/date dtype).
            print("Recomputing the correlation matrix after feature removal...")
            # Reuses the same sampled_df built above (it still has every original feature column -
            # only self.df was reassigned, sampled_df's lineage is untouched) instead of pulling the
            # full (now column-reduced) dataset to the driver again.
            remaining_pandas_df = sampled_df.select(*self.features).toPandas()
            new_corr_matrix = remaining_pandas_df.corr()

            # Save the new correlation matrix as an image
            new_corr_matrix_image_path = "new_correlation_matrix.png"
            print(f"Saving the new correlation matrix as an image to {new_corr_matrix_image_path}...")
            plt.figure(figsize=(12, 10))
            sns.heatmap(new_corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title("New Correlation Matrix After Feature Removal")
            plt.savefig(new_corr_matrix_image_path)
            plt.close()
            print("New correlation matrix image saved!")

        return self.df
    
    def get_removed_features(self) -> List[str]:
        """
        Get the list of removed features for the genetic algorithm.

        Returns:
        - List[str]: List of features that were removed during the selection process.
        """
        return self.features_to_remove

    def random_forest_feature_importance(self, top_n: int = 50, n_estimators: int = 100, random_state: int = 42, sample_fraction: float = 0.1, plot: bool = True) -> pd.DataFrame:
        """
        Compute and plot feature importance using a Random Forest Regressor, restricted to the
        'Finance' sector. This step only scores features - it does not remove any.

        Parameters:
        - top_n (int): Number of top features to display in the plot.
        - n_estimators (int): Number of trees in the RandomForestRegressor. Default 100.
        - random_state (int): Random seed for the RandomForestRegressor. Default 42.
        - sample_fraction (float): fraction of Finance-sector companies (fsym) sampled before
          converting to Pandas - avoids pulling every Finance-sector row to the driver. Default
          0.1 (10% of companies).

        Returns:
        - pd.DataFrame: DataFrame containing feature importance scores.
        """
        print("random_forest_feature_importance: starting...")
        rows_before, features_before = self._rows(), len(self.features)
        print(f"random_forest_feature_importance: input -> rows={rows_before}, features={features_before}")
        print(f"random_forest_feature_importance params: sector_filter='Finance', n_estimators={n_estimators}, "
              f"random_state={random_state}, top_n={top_n}, sample_fraction={sample_fraction}")

        # Filter the DataFrame for the "Finance" sector
        finance_df = self.df.filter(F.col('factset_sector_desc') == 'Finance')

        # Sample companies within Finance before converting to Pandas, rather than pulling every
        # Finance-sector row to the driver - mirrors the sampling in spearman_correlation().
        finance_companies = [row['fsym'] for row in finance_df.select('fsym').distinct().collect()]
        finance_sample_size = min(len(finance_companies), max(1, int(len(finance_companies) * sample_fraction)))
        sampled_finance_companies = random.sample(finance_companies, finance_sample_size)
        finance_df = finance_df.filter(F.col('fsym').isin(sampled_finance_companies))

        # Select features and target columns and convert to Pandas
        pandas_df = finance_df.select(*self.features, self.target).toPandas()

        X = pandas_df[self.features].copy()
        y = pandas_df[self.target].copy()

        rows_finance = len(pandas_df)
        print(f"Converted Spark DataFrame to Pandas DataFrame for Random Forest "
              f"({len(sampled_finance_companies)}/{len(finance_companies)} Finance companies sampled). "
              f"rows before sector filter={rows_before}, rows after filter to 'Finance'={rows_finance} "
              f"({rows_finance / rows_before:.1%} of rows kept; self.df itself is unaffected by this filter)")

        # Fill NA values
        X.fillna(0, inplace=True)
        y.fillna(0, inplace=True)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X_scaled, y)

        # Get feature importances
        importances = rf.feature_importances_
        importance_df = pd.DataFrame(importances, index=self.features, columns=['Importance'])

        # Sort by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        if plot:
            # Plotting the top N features
            top_importance_df = importance_df.head(top_n)

            plt.figure(figsize=(12, 8))
            sns.barplot(x=top_importance_df.index, y='Importance', data=top_importance_df)
            plt.xticks(rotation=90)
            plt.xlabel("Feature")
            plt.title(f'Top {top_n} Feature Importances using Random Forest Regressor')
            plt.savefig(f"top_{top_n}_rf_feature_importance.png", dpi=300, bbox_inches='tight')
            plt.show()

        print(f"random_forest_feature_importance: done. features unchanged at {features_before} (no removal at this step).")
        return importance_df

print("FeatureSelection class defined.")
