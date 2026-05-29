# Reflection

## What business question does the model answer?

The model answers a practical replenishment question: **how many units of each product should Grupo Bimbo deliver to each client, route, channel, and agency for a given week?**

In [04_improvements.ipynb](../experiments/04_improvements.ipynb), the target is `Demanda_uni_equil`, which represents adjusted demand after returns. The model is therefore not just predicting raw sales; it is trying to estimate the demand that should actually be satisfied after accounting for products that come back. In business terms, this supports better inventory allocation: fewer stockouts, fewer stale products, fewer unnecessary deliveries, and better planning across a very large retail distribution network.

The improved model uses XGBoost with lagged demand features, historical return-rate features, geospatial agency features, product metadata parsed from product names, and product-channel demand statistics. On the sampled validation setup in the notebook, RMSLE improves from `0.4646` to `0.4624`. The numerical gain is modest, but it shows that the model becomes stronger when it is given business-aware features rather than only raw IDs.

## Where does it fail, and what additional data would help?

The model is weakest when the past is not a good guide for the future. Cold-start cases are a major problem: a new client, product, agency, or product-client combination may have little or no lag history, which makes demand hard to estimate. It also has limited time coverage, so it cannot learn longer seasonal cycles, holidays, weather effects, paydays, school calendars, or promotion periods very well.

The geospatial data also has limits. The Kaggle data does not include true GPS coordinates, so agency locations are inferred from town and state names. Some agencies use town-level Nominatim coordinates, while others fall back to state centroids. Those fallback points preserve broad regional information, but they are not exact depot locations. This can weaken distance-based features and cluster assignments.

Additional data that would help includes true agency and client coordinates, route geometry, delivery frequency, inventory constraints, stockout indicators, product shelf life, prices, promotions, holidays, local weather, store type, store size, and richer customer metadata. These would help the model separate true low demand from missing supply, temporary promotion spikes, logistics constraints, or regional demand patterns.

## How does the geospatial dimension change the forecasting problem?

Geography turns the task from a purely tabular demand prediction problem into a spatial-temporal forecasting problem. Demand is not independent across rows: nearby agencies may share climate, income patterns, urban density, route structure, and product preferences. A product that performs well in one region may not behave the same way in another region, even if the historical global average looks similar.

The spatial features in the project, especially `geo_cluster` and `dist_to_hub_km`, give the model a way to learn regional structure. They help represent where an agency sits in the distribution network, not only what it sold last week. At the same time, geospatial features introduce a new responsibility: the model must know the quality of the location signal. A town-level match and a state-centroid fallback should not be treated as equally precise.

## One thing I genuinely learned

I learned that geospatial data can add useful business context even when it is imperfect, but it has to be handled carefully. Cleaning town names, caching geocoding results, tracking `geocode_source`, and creating conservative fallback coordinates mattered as much as the final map or cluster label.

I also learned how large the impact of feature engineering can be in traditional machine learning projects. The XGBoost model was important, but the real work was making the rows more meaningful: lagged demand, historical return behavior, product size and pack features, product-channel statistics, and agency geography all helped turn raw transactional IDs into signals the model could actually use. This project made it clear that, for structured business data, better features can matter more than a more complicated model.
