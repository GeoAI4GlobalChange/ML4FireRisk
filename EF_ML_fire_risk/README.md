# The hybrid ML code for modeling large fire probability
This reposiry provides the code of the hybrid fire model, used for predicting large fire probability in the western US.
## Run the model quickly
(1) Train, validate, and test the ML model:  
Run the 'ml_train_test_random_cross_validation_states' function in the 'Fire_Model_demo.py'  
(2) Detailed code of the three modules in the fire model:  
See the FF, FA, and HS modules in the 'model.py'  
## Data availability  
(1) Fire Program Analysis fire occurrence database: https://www.fs.usda.gov/rds/archive/Catalog/RDS-2013-0009.5, https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6  
(2) Monitoring trends and burn severity (MTBS): https://mtbs.gov/direct-download  
(3) Gridded surface meteorological data: https://www.climatologylab.org/gridmet.html  
(4) Topography data: https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1  
(5) MODIS land cover IGBP data: https://lpdaac.usgs.gov/products/mcd12q1v006/  
(6) MODIS Net Primary Production data: https://lpdaac.usgs.gov/products/myd17a3hv006/  
(7) Gridded Population data: https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-rev11  
(8) Fire Weather Index data: https://cds.climate.copernicus.eu/cdsapp#!/dataset/cems-fire-historical?tab=overview  
## References  
Please refer to the upcoming paper “Projecting large fires in the western US with an interpretable and accurate hybrid machine learning method” for more details. 
