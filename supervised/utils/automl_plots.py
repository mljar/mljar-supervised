import logging
import os
import traceback # For exception details

import numpy as np
import pandas as pd
import scipy as sp
# --- Added Import ---
from sklearn.preprocessing import MinMaxScaler
# --------------------

logger = logging.getLogger(__name__)
from supervised.utils.config import LOG_LEVEL
logger.setLevel(LOG_LEVEL)
# Add a handler if running standalone for testing
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class AutoMLPlots:
    # Original filename definitions
    features_heatmap_fname = "features_heatmap.png"
    correlation_heatmap_fname = "correlation_heatmap.png"
    # Filename for Scaled Plot
    features_heatmap_scaled_fname = "features_heatmap_scaled.png"

    @staticmethod
    def _plot_feature_heatmap(data_df, title, plot_path, cmap="Blues", vmin=None, vmax=None, cbar_label='Importance'):
        """ Helper method to generate and save a feature importance heatmap. """
        try:
            logger.info(f"Generating heatmap: '{title}'")
            # Adjust height dynamically based on number of features
            plot_height = max(7, len(data_df.index) * 0.35)
            fig, ax = plt.subplots(1, 1, figsize=(10, plot_height))

            image = ax.imshow(
                data_df,
                interpolation="nearest",
                cmap=plt.cm.get_cmap(cmap),
                aspect="auto",
                vmin=vmin, # Use provided vmin
                vmax=vmax  # Use provided vmax
            )
            cbar = plt.colorbar(mappable=image)
            cbar.set_label(cbar_label) # Use provided label

            x_tick_marks = np.arange(len(data_df.columns))
            y_tick_marks = np.arange(len(data_df.index))
            ax.set_xticks(x_tick_marks)
            ax.set_xticklabels(data_df.columns, rotation=90)
            ax.set_yticks(y_tick_marks)
            ax.set_yticklabels(data_df.index)
            ax.set_title(title)

            plt.tight_layout(pad=2.0)
            plt.savefig(plot_path)
            logger.info(f"Saved heatmap to: {plot_path}")
            plt.close(fig) # Close the specific figure

        except Exception as e:
            logger.error(f"Failed to generate heatmap '{title}': {e}")
            logger.error(traceback.format_exc())
            plt.close("all") # Close any potentially open plots on error


    @staticmethod
    def add(results_path, models, fout):
        """
        Adds plots to the report file stream. Now includes both unscaled and scaled importance.

        Args:
            results_path (str): Path to results directory.
            models (list): List of model objects.
            fout (file object): Writable file object for the report.
        """
        # Generate both feature importance plots
        AutoMLPlots.models_feature_importance(results_path, models)

        # --- Unscaled Feature Importance Section ---
        features_plot_path = os.path.join(
            results_path, AutoMLPlots.features_heatmap_fname # Use original filename
        )
        if os.path.exists(features_plot_path):
            fout.write("\n\n### Features Importance (Original Scale)\n") # Updated title
            fout.write(
                f"![features importance across models]({AutoMLPlots.features_heatmap_fname})\n\n" # Use original filename
            )
        else:
            logger.warning(f"Original feature importance plot not found at: {features_plot_path}")


        # --- Scaled Feature Importance Section ---
        features_scaled_plot_path = os.path.join(
            results_path, AutoMLPlots.features_heatmap_scaled_fname # Use scaled filename
        )
        if os.path.exists(features_scaled_plot_path):
            fout.write("\n\n### Scaled Features Importance (MinMax per Model)\n") # Title for scaled plot
            fout.write(
                f"![scaled features importance across models]({AutoMLPlots.features_heatmap_scaled_fname})\n\n" # Use scaled filename
            )
        else:
            logger.warning(f"Scaled feature importance plot not found at: {features_scaled_plot_path}")


        # --- Correlation Section (remains the same) ---
        AutoMLPlots.models_correlation(results_path, models)

        correlation_plot_path = os.path.join(
            results_path, AutoMLPlots.correlation_heatmap_fname
        )
        if os.path.exists(correlation_plot_path):
            fout.write("\n\n### Spearman Correlation of Models\n")
            fout.write(
                f"![models spearman correlation]({AutoMLPlots.correlation_heatmap_fname})\n\n"
            )
        else:
            logger.warning(f"Model correlation plot not found at: {correlation_plot_path}")


    @staticmethod
    def models_feature_importance(results_path, models):
        """
        Generates and saves BOTH original and scaled feature importance heatmaps.
        """
        logger.info("Starting feature importance generation (original and scaled).")
        try:
            # --- Data Aggregation (Common part) ---
            model_feature_imp = {}
            # (Same robust reading logic as before)
            for m in models:
                model_name = m.get_name()
                model_path = os.path.join(results_path, model_name)
                logger.debug(f"Processing model '{model_name}' in '{model_path}'")
                if not os.path.isdir(model_path):
                    logger.warning(f"Directory not found for model '{model_name}'. Skipping.")
                    continue
                try:
                    all_files = os.listdir(model_path)
                except OSError as e:
                    logger.error(f"Cannot list directory {model_path}: {e}. Skipping model '{model_name}'.")
                    continue
                imp_data = [f for f in all_files if "_importance.csv" in f and "shap" not in f]
                if not imp_data:
                    logger.warning(f"No suitable importance files found for model '{model_name}'. Skipping.")
                    continue
                df_all = []
                for fname in imp_data:
                    file_path = os.path.join(model_path, fname)
                    try:
                        df = pd.read_csv(file_path, index_col=0)
                        numeric_df = df.select_dtypes(include=np.number)
                        if numeric_df.empty or numeric_df.isnull().all().all():
                            logger.warning(f"File {fname} (model '{model_name}') contains no valid numeric data. Skipping.")
                            continue
                        df_all.append(df)
                    except Exception as read_e:
                        logger.error(f"Error reading/processing file {fname} (model '{model_name}'): {read_e}. Skipping.")
                        continue
                if not df_all:
                    logger.warning(f"No valid importance dataframes read for model '{model_name}'. Skipping.")
                    continue
                try:
                    df_concat = pd.concat(df_all, axis=1, join='outer')
                    numeric_df_concat = df_concat.select_dtypes(include=np.number)
                    if not numeric_df_concat.empty:
                         model_feature_imp[model_name] = numeric_df_concat.mean(axis=1).fillna(0)
                    else:
                         logger.warning(f"No numeric data after concat for model '{model_name}'. Skipping.")
                except Exception as concat_e:
                     logger.error(f"Error aggregating importance for model '{model_name}': {concat_e}")
                     continue

            logger.info(f"Collected feature importance for {len(model_feature_imp)} models.")
            if len(model_feature_imp) < 2:
                logger.warning("Feature importance heatmaps require at least 2 models with data. Skipping plot generation.")
                return

            mfi = pd.concat(model_feature_imp, axis=1, join='outer').fillna(0)
            logger.debug(f"Combined importance DataFrame shape: {mfi.shape}")

            # --- Sorting & Top N (Common part) ---
            mfi["m"] = mfi.mean(axis=1)
            mfi_sorted = mfi.sort_values(by="m", ascending=False)
            mfi_sorted = mfi_sorted.drop("m", axis=1) # Keep original mfi for potential later use if needed

            num_features_original = mfi_sorted.shape[0]
            mfi_plot_data = mfi_sorted # Default to using all sorted features
            title_suffix = "Feature Importance"
            scaled_title_suffix = "Scaled Feature Importance (MinMax per model)"

            if num_features_original > 25:
                mfi_plot_data = mfi_sorted.head(25)
                title_suffix = f"Top-25 ({num_features_original} total) Feature Importance"
                scaled_title_suffix = f"Top-25 ({num_features_original} total) Scaled Feature Importance (MinMax per model)"
                logger.info(f"Selecting top 25 features out of {num_features_original} for plotting.")
            else:
                 logger.info(f"Using all {num_features_original} features for plotting.")


            # --- Plotting Unscaled Version ---
            unscaled_plot_path = os.path.join(results_path, AutoMLPlots.features_heatmap_fname)
            AutoMLPlots._plot_feature_heatmap(
                data_df=mfi_plot_data,
                title=title_suffix + " (Original Scale)",
                plot_path=unscaled_plot_path,
                cbar_label='Importance'
                # vmin/vmax are auto-detected by default
            )

            # --- Scaling Data ---
            logger.debug("Applying Min-Max scaling for the second plot.")
            scaler = MinMaxScaler()
            mfi_scaled_array = scaler.fit_transform(mfi_plot_data) # Scale the potentially filtered data
            mfi_scaled = pd.DataFrame(mfi_scaled_array, index=mfi_plot_data.index, columns=mfi_plot_data.columns)

            # --- Plotting Scaled Version ---
            scaled_plot_path = os.path.join(results_path, AutoMLPlots.features_heatmap_scaled_fname)
            AutoMLPlots._plot_feature_heatmap(
                data_df=mfi_scaled,
                title=scaled_title_suffix,
                plot_path=scaled_plot_path,
                vmin=0, # Explicit range for scaled data
                vmax=1,
                cbar_label='Scaled Importance (MinMax per model)'
            )

            logger.info("Finished generating feature importance plots.")

        except Exception as e:
            logger.error(f"An error occurred during feature importance processing: {e}")
            logger.error(traceback.format_exc())
            plt.close("all") # Ensure plots are closed on unexpected error


    # --- correlation and models_correlation methods remain the same as in the previous version ---
    # (Include the improved versions from the previous response here)
    @staticmethod
    def correlation(oof1, oof2):
        """ Calculates mean Spearman correlation between prediction columns """
        # (Original code - unchanged)
        cols = [c for c in oof1.columns if "prediction" in c]
        # Check if prediction columns exist
        if not cols or not all(c in oof2.columns for c in cols):
            logger.warning("Prediction columns mismatch or not found for correlation calculation.")
            return np.nan # Return NaN if predictions can't be compared

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore")
            v = []
            for c in cols:
                try:
                    # Calculate Spearman correlation, ignore p-value
                    corr_val, _ = sp.stats.spearmanr(oof1[c], oof2[c])
                    # Handle potential NaN result from spearmanr if input variance is zero
                    if not np.isnan(corr_val):
                        v.append(corr_val)
                    else:
                         logger.debug(f"NaN result from spearmanr for column {c}. Skipping.")
                except Exception as corr_e:
                    logger.warning(f"Could not calculate Spearman correlation for column {c}: {corr_e}")

        # Return mean correlation, or NaN if no valid correlations were calculated
        return np.mean(v) if v else np.nan


    @staticmethod
    def models_correlation(results_path, models):
        """ Generates and saves model prediction correlation heatmap """
        # (Original code - minor logging/error handling improvements)
        logger.info("Starting model correlation heatmap generation.")
        try:
            if len(models) < 2:
                logger.warning("Model correlation heatmap requires at least 2 models. Skipping.")
                return

            names = []
            oofs = []
            valid_models_indices = [] # Keep track of models with valid OOF data

            for i, m in enumerate(models):
                try:
                    oof_data = m.get_out_of_folds()
                    # Basic validation of OOF data
                    if oof_data is None or oof_data.empty or not any("prediction" in c for c in oof_data.columns):
                        logger.warning(f"Model '{m.get_name()}' has invalid or missing out-of-folds prediction data. Excluding from correlation.")
                        continue

                    names.append(m.get_name())
                    oofs.append(oof_data)
                    valid_models_indices.append(i) # Store original index if valid
                    logger.debug(f"Got valid OOF data for model '{m.get_name()}'.")

                except AttributeError:
                     logger.warning(f"Model '{m.get_name()}' seems to be missing 'get_out_of_folds' method or it failed. Excluding from correlation.")
                     continue
                except Exception as oof_e:
                     logger.warning(f"Failed to get OOF data for model '{m.get_name()}': {oof_e}. Excluding from correlation.")
                     continue


            num_valid_models = len(names)
            if num_valid_models < 2:
                logger.warning(f"Fewer than 2 models ({num_valid_models}) have valid OOF data for correlation. Skipping plot generation.")
                return

            logger.info(f"Calculating correlations for {num_valid_models} models.")
            corrs = np.ones((num_valid_models, num_valid_models)) # Use num_valid_models dimension
            for i in range(num_valid_models):
                for j in range(i + 1, num_valid_models):
                    correlation_value = AutoMLPlots.correlation(oofs[i], oofs[j])
                    # Fill with NaN if correlation calculation failed
                    corrs[i, j] = corrs[j, i] = correlation_value if not np.isnan(correlation_value) else np.nan


            # Check if all correlations are NaN
            if np.isnan(corrs[np.triu_indices(num_valid_models, k=1)]).all():
                 logger.warning("All pairwise model correlations resulted in NaN. Cannot generate heatmap.")
                 return


            logger.info("Generating model correlation heatmap.")
            figsize = (15, 15) if num_valid_models > 15 else (10, 10) # Adjusted threshold
            fig, ax = plt.subplots(1, 1, figsize=figsize)

            image = ax.imshow(
                corrs,
                interpolation="nearest",
                cmap=plt.cm.get_cmap("Blues"),
                aspect="auto",
                vmin=np.nanmin(corrs), # Use nanmin/nanmax to handle potential NaNs
                vmax=np.nanmax(corrs)
            )
            plt.colorbar(mappable=image)

            x_tick_marks = np.arange(num_valid_models)
            y_tick_marks = np.arange(num_valid_models)
            ax.set_xticks(x_tick_marks)
            ax.set_xticklabels(names, rotation=90)
            ax.set_yticks(y_tick_marks)
            ax.set_yticklabels(names)
            ax.set_title("Spearman Correlation of Models' OOF Predictions") # Slightly more descriptive title

            plt.tight_layout(pad=2.0)

            # --- Saving the Plot ---
            os.makedirs(results_path, exist_ok=True) # Ensure directory exists
            plot_path = os.path.join(
                results_path, AutoMLPlots.correlation_heatmap_fname
            )
            plt.savefig(plot_path)
            logger.info(f"Saved model correlation heatmap to: {plot_path}")
            plt.close("all") # Close plot to free memory

        except Exception as e:
            # Log the exception with traceback
            logger.error(f"An error occurred during model correlation plotting: {e}")
            logger.error(traceback.format_exc())
             # Ensure plot is closed if error occurred during saving/closing
            plt.close("all")


