library(plyr)
library(mltools)
library(data.table)
library(tidyverse)
library(qdapTools)
library(stringr)
library(ggplot2)

setwd("C:/MSCDISS")
data <- read.csv("Data_Entry_2017.csv")



data <- data %>% select(1,2,4,5,6)

mdata <- data %>% filter(Patient.Gender == "M")

mdataage <- mdata$Patient.Age

sd(mdataage)

Fdata <- data %>% filter(Patient.Gender == "F")

Fdataage <- Fdata$Patient.Age

sd(Fdataage)


data$Finding.Labels <-gsub("No Finding", "NoFinding", data$Finding.Labels)

gender <- ggplot(data = data, mapping = aes(x= Patient.Gender, fill = Patient.Gender)) +
  geom_bar(stat='count') + xlab("Gender") + ylab("Number of people") +
  scale_fill_discrete(name= "Gender") +
  geom_text(stat='count', aes(label=..count..), vjust=-1)

gender 

disease_count_data <- data %>%
  separate_rows(Finding.Labels) %>%
  count(Finding.Labels)

g <- ggplot(data = disease_count_data, mapping = aes(x= n, y = Finding.Labels, fill = Finding.Labels)) +
  geom_bar(stat="identity") +
  xlab("count of disease instances") + 
  ylab("Finding Label") +
  scale_fill_discrete(name= "Diseases") +
  geom_text(aes(label=n))
  

g



disease_gender_data <- data %>%
  separate_rows(Finding.Labels) %>%
  group_by(Patient.Gender) %>%
  count(Finding.Labels)

b <- ggplot(data = disease_gender_data, mapping = aes(x= n, y = Finding.Labels, fill = Patient.Gender)) +
  geom_bar(stat="identity") +
  xlab("count of disease instances") +
  ylab("Finding Label") +
  scale_fill_discrete(name= "Gender", breaks=c("F", "M"), labels=c("Female", "Male")) +
  geom_text(aes(label=n), position= position_stack(vjust = 0.5))
  
b 



labs <- c(paste(seq(0, 95, by = 5), seq(0 + 5 - 1, 100 - 1, by = 5),
                sep = "-"), paste(100, "+", sep = ""))

data$AgeGroup <- cut(data$Patient.Age, breaks = c(seq(0, 100, by = 5), Inf), labels = labs, right = FALSE)


age_graph <- ggplot(data = data, mapping = aes(x= AgeGroup, color= AgeGroup, fill= AgeGroup)) +
  geom_histogram(stat="count") + xlab("Age groups")

age_graph


disease_age_finding_data <- data %>%
  separate_rows(Finding.Labels) %>%
  group_by(AgeGroup) %>%
  count(Finding.Labels)

age_finding_graph <- ggplot(data = disease_age_finding_data, mapping = aes(x= n, y = Finding.Labels, fill = AgeGroup)) +
  geom_bar(stat="identity") + xlab("count of disease instances") + ylab("Finding Label")

age_finding_graph


ResNet_data <- read.csv("ResNet150V2_output.csv")

library("tidyverse")
df <- ResNet_data %>%
  select(epoch, AUC, loss, precision, recall) %>%
  gather(key = "variable", value = "value", -epoch)


ggplot(df, aes(x = epoch, y = value)) + 
  geom_line(aes(color = variable), size =1) + 
  scale_color_manual(values = c("red", "blue", "Green", "black"))+
 theme_dark()


