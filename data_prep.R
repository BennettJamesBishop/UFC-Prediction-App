library(readr)
library(dplyr)

# Load your original dataset
ufc_original_dataset <- read_csv('ufc_dataset.csv')

# Select Predictors
ufc_data <- ufc_original_dataset %>%
    select(winner, weight_class,
         r_fighter, b_fighter, 
         r_wins_total, b_wins_total, 
         r_losses_total, b_losses_total,
         r_age, b_age, 
         r_height, b_height, 
         r_weight, b_weight, 
         r_reach, b_reach, 
         r_stance, b_stance,
         r_SLpM_total, b_SLpM_total, 
         r_SApM_total, b_SApM_total,
         r_sig_str_acc_total, b_sig_str_acc_total,
         r_td_acc_total, b_td_acc_total,
         r_str_def_total, b_str_def_total,
         r_td_def_total, b_td_def_total,
         r_sub_avg, b_sub_avg, 
         r_td_avg, b_td_avg)


#Fix weight classes
# Drop rows where weight_class contains numbers 1 to 30 (UFC Eventd 1-30)
pattern <- "\\b(1[0-9]|2[0-9]|30|[1-9])\\b"
ufc_data <- ufc_data[!grepl(pattern, ufc_data$weight_class), ]

# Drop rows where weight_class represents a pre-2001 fight, or fights where occured at a super rare weight class
drop_values <- c("Ultimate Japan Heavyweight Tournament Title", 
                 "Ultimate Ultimate '95 Tournament Title", 
                 "Ultimate Ultimate '96 Tournament Title",
                 "UFC Superfight Championship",
                "Super Heavyweight",
                "Open Weight",
                "Catch Weight")

ufc_data <- ufc_data[!ufc_data$weight_class %in% drop_values, ]  

ufc_data <- ufc_data %>% 
  mutate(weight_class = case_when(
     weight_class == "UFC Bantamweight Title" ~ "Bantamweight",
     weight_class == "UFC Featherweight Title" ~ "Featherweight",
      weight_class == "UFC Flyweight Title" ~ "Flyweight",
     weight_class == "UFC Heavyweight Title" ~ "Heavyweight",
      weight_class == "UFC Interim Bantamweight Title" ~ "Bantamweight",
     weight_class == "UFC Interim Featherweight Title" ~ "Featherweight",
      weight_class == "UFC Interim Flyweight Title" ~ "Flyweight",
     weight_class == "UFC Interim Heavyweight Title" ~ "Heavyweight",
         weight_class == "UFC Interim Light Heavyweight Title" ~ "Light Heavyweight",
         weight_class == "UFC Interim Lightweight Title" ~ "Lightweight",
         weight_class == "UFC Interim Middleweight Title" ~ "Middleweight",
         weight_class == "UFC Interim Welterweight Title" ~ "Welterweight",
         weight_class == "UFC Light Heavyweight Title" ~ "Light Heavyweight",
         weight_class == "UFC Lightweight Title" ~ "Lightweight",
         weight_class == "UFC Middleweight Title" ~ "Middleweight",
        weight_class == "UFC Welterweight Title" ~ "Welterweight",
        weight_class == "UFC Women's Bantamweight Title" ~ "Women's Bantamweight",
        weight_class == "UFC Women's Featherweight Title" ~ "Women's Featherweight",
      weight_class == "UFC Women's Flyweight Title" ~ "Women's Flyweight",
      weight_class == "UFC Women's Strawweight Title" ~ "Women's Strawweight",
        weight_class == "TUF Nations Canada vs. Australia Middleweight Tournament Title" ~ "Middleweight",
      weight_class == "TUF Nations Canada vs. Australia Welterweight Tournament Title" ~ "Welterweight",
      weight_class == "Ultimate Fighter Australia vs. UK Lightweight Tournament Title" ~ "Lightweight",
      weight_class == "Ultimate Fighter Australia vs. UK Welterweight Tournament Title" ~ "Welterweight",
      weight_class == "Ultimate Fighter China Featherweight Tournament Title" ~ "Featherweight",
      weight_class == "Ultimate Fighter China Welterweight Tournament Title" ~ "Welterweight",
     weight_class == "Ultimate Fighter Latin America Bantamweight Tournament Title" ~ "Bantamweight",
      weight_class == "Ultimate Fighter Latin America Featherweight Tournament Title" ~ "Featherweight",
    .default = weight_class
  ))


#Create matchup_data, which has new 'difference' variables, calculated as 'RED - BLUE'
matchup_data <- ufc_data %>%
   mutate(
    career_wins_diff = r_wins_total - b_wins_total,
    career_losses_diff = r_losses_total - b_losses_total,
    height_diff = r_height - b_height,
    weight_diff = r_weight - b_weight,
    reach_diff = r_reach - b_reach,
    SLpM_diff = r_SLpM_total - b_SLpM_total,
    SApM_diff = r_SApM_total - b_SApM_total,
    sig_str_acc_diff = r_sig_str_acc_total - b_sig_str_acc_total,
    td_acc_diff = r_td_acc_total - b_td_acc_total,
    str_def_diff = r_str_def_total - b_str_def_total,
    td_def_diff = r_td_def_total - b_td_def_total,
    sub_avg_diff = r_sub_avg - b_sub_avg,
    td_avg_diff = r_td_avg - b_td_avg
  ) %>%
  select(winner, weight_class, r_age, b_age, r_stance, b_stance, 
         career_wins_diff, career_losses_diff, height_diff, weight_diff, reach_diff, 
         SLpM_diff, SApM_diff, 
         sig_str_acc_diff, td_acc_diff,
         str_def_diff, td_def_diff,
         sub_avg_diff, td_avg_diff)



library(dplyr)
library(readr)

# Load UFC Matches dataset
ufc_original_dataset <- read_csv('ufc_dataset.csv')

# Select Predictors
ufc_data <- ufc_original_dataset %>%
  select(winner, weight_class,
         r_fighter, b_fighter, 
         r_wins_total, b_wins_total, 
         r_losses_total, b_losses_total,
         r_age, b_age, 
         r_height, b_height, 
         r_weight, b_weight, 
         r_reach, b_reach, 
         r_stance, b_stance,
         r_SLpM_total, b_SLpM_total, 
         r_SApM_total, b_SApM_total,
         r_sig_str_acc_total, b_sig_str_acc_total,
         r_td_acc_total, b_td_acc_total,
         r_str_def_total, b_str_def_total,
         r_td_def_total, b_td_def_total,
         r_sub_avg, b_sub_avg, 
         r_td_avg, b_td_avg)


#Create matchup_data, which has new 'difference' variables, calculated as 'RED - BLUE'
matchup_data <- ufc_data %>%
  mutate(
    career_wins_diff = r_wins_total - b_wins_total,
    career_losses_diff = r_losses_total - b_losses_total,
    age_diff = r_age - b_age,
    height_diff = r_height - b_height,
    weight_diff = r_weight - b_weight,
    reach_diff = r_reach - b_reach,
    SLpM_diff = r_SLpM_total - b_SLpM_total,
    SApM_diff = r_SApM_total - b_SApM_total,
    sig_str_acc_diff = r_sig_str_acc_total - b_sig_str_acc_total,
    td_acc_diff = r_td_acc_total - b_td_acc_total,
    str_def_diff = r_str_def_total - b_str_def_total,
    td_def_diff = r_td_def_total - b_td_def_total,
    sub_avg_diff = r_sub_avg - b_sub_avg,
    td_avg_diff = r_td_avg - b_td_avg
  ) %>%
  select(winner, r_stance, b_stance, 
         career_wins_diff, career_losses_diff, age_diff, height_diff, weight_diff, reach_diff, 
         SLpM_diff, SApM_diff, 
         sig_str_acc_diff, td_acc_diff,
         str_def_diff, td_def_diff,
         sub_avg_diff, td_avg_diff)

#Write matchup_data to csv file
write_csv(matchup_data, 'matchup_data.csv')


# Load Fighter Stats dataset
fighter_stats <- read_csv('fighter_stats.csv')

# Check for remaining missing values
sapply(fighter_stats, function(x) sum(is.na(x)))

# Remove row where 'name' is missing (removes the empty row)
fighter_stats <- fighter_stats %>%
  filter(!is.na(name))

# Impute remaining missing values (weight, reach, stance)
fighter_stats <- fighter_stats %>%
  mutate(
    reach = ifelse(is.na(reach), height, reach),
    age = ifelse(is.na(age), median(age, na.rm = TRUE), age),
    stance = ifelse(is.na(stance), 'Orthodox', stance)  # Impute missing stance with 'Orthodox'
  )

#Write fighter_stats to csv file
write_csv(fighter_stats, 'fighter_stats.csv')

