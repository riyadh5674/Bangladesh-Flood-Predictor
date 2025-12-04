# Bangladesh-Flood-Predictor
Physics-guided LightGBM flood predictor for Bangladesh. 0.95+ AUC using river/coast distance, elevation-aware spatial risk, monsoon + climate-trend seasonal_risk &amp; monotonic constraints. Full notebook, data &amp; model. Ready for early-warning systems. #flood #bangladesh #physics-ml

Here are all the all the features of the data i am working on:

| features          | units             |
| ----------------- | ----------------- |
| Sl(serial)        |                   |
| Station Name      |                   |
| Year              |                   |
| Month             |                   |
| Max_temp          | celsius           |
| Min_temp          | celsius           |
| Rainfall          | cm                |
| Relative Humidity | percentage        |
| Wind Speed        | meters per second |
| Cloud Coverage    | okta              |
| Bright Sunshine   | Hours per day     |
| Station Number    |                   |
| X_coordinate      |                   |
| Y_coordinate      |                   |
| Lattitude         |                   |
| longitude         |                   |
| altitude          | meter             |
| flood?            | 1(Yes) & 0(No)    |

**Step 1**
the first step is to select features , the following features are selected to work on this project:

| features          | units             |
| ----------------- | ----------------- |
| Year              |                   |
| Month             |                   |
| Max_temp          | celsius           |
| Min_temp          | celsius           |
| Rainfall          | cm                |
| Relative Humidity | percentage        |
| Cloud Coverage    | okta              |
| Bright Sunshine   | Hours per day     |
| X_coordinate      |                   |
| Y_coordinate      |                   |
| Lattitude         |                   |
| longitude         |                   |
| altitude          | meter             |
| flood?            | 1(Yes) & 0(No)    |

**Step 2**

The following table helps you understand the relationship between feature and the probability of flood . 

| Feature           | If value increases ↑         | Effect on Flood Risk | Hydrological / Physical Reason                                                                 |
| ----------------- | ---------------------------- | -------------------- | ---------------------------------------------------------------------------------------------- |
| Max_Temp          | Higher maximum temperature   | ↓ Usually decreases  | Increases evaporation → less soil moisture and runoff                                          |
| Min_Temp          | Higher minimum temperature   | ↓ Slightly decreases | Warmer nights reduce frost, increase evaporation                                               |
| Rainfall          | More rainfall                | ↑ Strongly increases | Direct water input; exceeds infiltration & drainage capacity → surface runoff & river overflow |
| Relative_Humidity | Higher humidity              | ↑ Increases          | Reduces evaporation → soil stays saturated longer, more runoff during rain                     |
| Cloud_Coverage    | More cloud cover             | ↑ Increases          | Indicates prolonged rain possibility, blocks sunshine → less evaporation                       |
| Bright_Sunshine   | More sunshine hours          | ↓ Decreases          | Higher evaporation and soil drying → lower flood probability                                   |
| ALT (elevation)   | Higher altitude              | ↓ Strongly decreases | Locations above floodplains, faster runoff, less water accumulation                            |
| Month             | Monsoon/post-monsoon months  | ↑ Increases          | Seasonal peak rainfall (e.g., Jun–Sep in South Asia)                                           |
| Year              | More recent years            | ↑ Slightly increases | Climate change → more intense & frequent extreme rainfall events                               |
| Period            | Longer/later rainfall period | ↑ Increases          | Cumulative rainfall leads to soil saturation and higher runoff                                 |

>These features are directly related to flood probability  


**Step 3**
**Minimum viable physics-guided spatial features:**

1. **Distance to nearest river** (from X_COR/Y_COR or LAT/LONG) → strongest single feature
2. **Distance to nearest coastline** (only if your stations are <150–200 km from sea)
3. **Interaction term**: risk_spatial = log(1 / (distance_to_river_m + 100)) × (1 / (ALT + 10)) → being close to river AND low elevation = extremely high risk (this one interaction often beats 10 other features)


i created new features using #geopandas . the new columns with their relationship with flood probability is shown below:


| Feature              | If value increases ↑ | Consequence |
| -------------------- | -------------------- | ----------- |
| dist_to_river_km     |                      | ↓ Decreases |
| river_proximity_risk |                      | ↑ Increases |
| spatial_flood_risk   |                      | ↑ Increases |
| dist_to_coast_km     |                      | ↓ Decreases |
| coastal_risk         |                      | ↑ Increases |

**Step 4**
Understanding seasonal data , 
Bangladesh flood pattern is crystal clear:

- **Peak flood months**: June–October (especially July–September)
- **Pre-monsoon**: May–June → sudden heavy rain + soil not yet saturated
- **Post-monsoon**: October–November → water recedes slowly
- **Dry season**: December–April → almost zero flood risk
- Recent years (2017, 2019, 2020, 2022, 2024) = much worse floods due to climate change

we created new feature - "seasonal_risk"

**Step 5**

Now we have all the features that is directly related to the probability of flood :

| Feature              | If value increases ↑ | Consequence |
| -------------------- | -------------------- | ----------- |
| Rainfall             |                      | ↑ Increases |
| max_temp             |                      | ↓ Decreases |
| min_temp             |                      | ↓ Decreases |
| Relative Humidity    |                      | ↑ Increases |
| Cloud_Coverage       |                      | ↑ Increases |
| Bright sunshine      |                      | ↓ Decreases |
| ALT                  |                      | ↓ Decreases |
| river_proximity_risk |                      | ↑ Increases |
| spatial_flood_risk   |                      | ↑ Increases |
| coastal_risk         |                      | ↑ Increases |
| seasonal_risk        |                      | ↑ Increases |
| period               |                      | ↑ Increases |

**Step 6**

Now we will preprocess data , this is pretty much clean except the flood? column has null values , we asigned the null values to be 0(No flood). and then train and split data , we will move to train the data with the following machine learning model :
#LIGHTGBM :
we trained a model using these feature , and then used the same model but physics guided , the results are :
**BASELINE AUC : 0.9965** 
**PHYSICS-GUIDED AUC: 0.8889** 
**IMPROVEMENT : -0.1077**

#CATBOOST :
 I ran this model and got good results :
**CATBOOST PHYSICS-GUIDED AUC: 0.9848**


**FINAL FINDINGS**
>Unconstrained LightGBM → 0.9964 AUC (random split) → heavily overfitted(Not trustworthy)

>Physics-guided LightGBM → stuck at ~0.88–0.90 with honest monotonic constraints

>Physics-guided CatBoost (final model) → 0.9848 AUC with real monotonic constraints

>Gap only 0.0116 while being 100% physically consistent


## Data Source & Citation

This project uses the publicly available **Bangladesh flood dataset** originally collected and published by:

**Noushin Gauhar, Sunanda Das, and Khadiza Sarwar Moury**  
in their 2021 IEEE paper:

> Gauhar, N., Das, S., & Moury, K. S. (2021). Prediction of Flood in Bangladesh using k-Nearest Neighbors Algorithm. In *2021 2nd International Conference on Robotics, Electrical and Signal Processing Techniques (ICREST)* (pp. 357–361). IEEE.  
> DOI: [10.1109/ICREST51555.2021.9331123](https://doi.org/10.1109/ICREST51555.2021.9331123)

GitHub repository:
https://github.com/n-gauhar/Flood-prediction


We sincerely thank the authors for generously making their high-quality dataset publicly available on GitHub. This work builds directly on their data collection efforts and would not have been possible without their contribution.

**If you use this dataset or our repository, please cite their original paper:**

```bibtex
@inproceedings{gauhar2021prediction,
  title={Prediction of Flood in Bangladesh using k-Nearest Neighbors Algorithm},
  author={Gauhar, Noushin and Das, Sunanda and Moury, Khadiza Sarwar},
  booktitle={2021 2nd International Conference on Robotics, Electrical and Signal Processing Techniques (ICREST)},
  pages={357--361},
  year={2021},
  organization={IEEE},
  doi={10.1109/ICREST51555.2021.9331123}
}



