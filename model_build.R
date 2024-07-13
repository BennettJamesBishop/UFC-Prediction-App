library(tidyverse)   # using tidyverse and tidymodels for this project mostly
library(tidymodels)
library(janitor)     # for cleaning out our data
library(randomForest)   # for building our randomForest
library(stringr)    # for matching strings
library("dplyr")     # for basic r functions
library("yardstick") # for measuring certain metrics
tidymodels_prefer()

#Read in matchup dataset created in data_prep file
matchup_data <- read_csv('matchup_data.csv')

#Test/Train Split
set.seed(4)  # setting a seed so the split is the same
matchup_data <- matchup_data %>% 
  mutate(winner = factor(winner),
         r_stance = factor(r_stance),
         b_stance = factor(b_stance))
matchup_split <- matchup_data %>%
  initial_split(prop = 0.8, strata = "winner")

matchup_train <- training(matchup_split) # training split
matchup_test <- testing(matchup_split) # testing split


#Making Recipe (Saved at bottom of file)
matchup_recipe <-
  # Recipe for the 'winner' variable using our training data
  recipe(winner ~ ., data = matchup_train) %>% 
  # Estimate any missing values of reach difference using that row's weight class and height difference values
  step_impute_linear(reach_diff,  impute_with = imp_vars(height_diff)) %>%
  # Estimate any missing values of age difference using the median age
  step_impute_median(age_diff) %>%
  # Replace any missing stance values with the most commonly occurring value of stance
  step_impute_mode(r_stance, b_stance) %>%
  # Create a new level for any unseen factor levels in the nominal predictors
  step_novel(all_nominal_predictors()) %>% 
  # Convert all nominal predictors to dummy variables
  step_dummy(all_nominal_predictors()) %>% 
  # Remove predictors with near-zero variance
  step_nzv(all_predictors()) %>% 
  # Normalize all predictors
  step_normalize(all_predictors())

# Prepare the recipe and apply it to the training data
prep(matchup_recipe) %>% bake(matchup_train)

#K-Fold Cross Validation
matchup_folds <- vfold_cv(matchup_train, v = 10, strata = winner)


# Create RF Model
rf_matchup_spec <- rand_forest(mtry = tune(), 
                               trees = tune(), 
                               min_n = tune()) %>%
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

rf_matchup_wf <- workflow() %>% 
  add_model(rf_matchup_spec) %>% 
  add_recipe(matchup_recipe)

rf_grid <- grid_regular(mtry(range = c(1, 6)), 
                        trees(range = c(200, 600)),
                        min_n(range = c(10, 20)),
                        levels = 5)

#Commenting out so I do not accidently run this (takes 20 mins)
#Creating grid of hyperparameter combos
 tune_rf <- tune_grid(
   rf_matchup_wf,
   resamples = matchup_folds,
   grid = rf_grid,
   control = control_grid(verbose = TRUE)
)


#Select the best hyperparameters based on ROC AUC
best_matchup_rf <- select_best(tune_rf, metric='roc_auc')

#Finalize the workflow with the best hyperparameters (0.764 ROC, 0.724 Accuracy)
final_rf_model <- finalize_workflow(rf_matchup_wf, best_matchup_rf)

#Save necessary stuff
saveRDS(matchup_recipe, file = "prep-RDA/matchup_recipe.rds")
saveRDS(tune_rf, file = "prep-RDA/tune_rf.rds")
saveRDS(final_rf_model, file = "prep-RDA/final_rf_model.rds")


#To evaluate model performance (fit the finalized model using cross-validation):
# final_rf_fit <- fit_resamples(final_rf_model, matchup_folds)
# metrics <- collect_metrics(final_rf_fit)
# print(metrics)