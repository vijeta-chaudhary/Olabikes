#!/usr/bin/env python
# coding: utf-8

# In[19]:


get_ipython().system('pip install gpxpy')


# In[20]:


#Data Connection

storage_account_name = "ola5132340602"
storage_account_access_key = "DJ6N0bGIFOCRZCIJR/fD+4+yA0URX8jKqyHI5MLBX2Ikva/X0eyXTINPCm0OKC2tNSwme1n1gqi0+ASt4k4zzw=="
container_name = "azureml-blobstore-c20a3cf7-1351-4b9e-b08a-e2dfe7d33b91"
file_path = "preprocessedFinal.csv"

# Set the Azure Storage account key in Spark configuration
spark.conf.set(
    "fs.azure.account.key." + storage_account_name + ".blob.core.windows.net",
    storage_account_access_key
)

# Construct the full file location
file_location = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/{file_path}"

# Read the CSV file from Azure Storage with UTF-8 encoding
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").option("encoding", "UTF-8").load(file_location)


# In[21]:


import pandas as pd 
import numpy as np


# In[22]:


df.show()


# In[23]:


coord = df.select("pick_lat", "pick_lng").collect() #saving the pick-up latitude and longitude in coord 
coord_np = np.array(coord) # converting coord to numpy array
neighbors = []


# In[24]:


from sklearn.cluster import MiniBatchKMeans
import gpxpy
import numpy as np

def min_distance(regionCenters, totalClusters):
    good_points = 0
    bad_points = 0
    less_dist = []
    more_dist = []
    min_distance = np.inf  #any big number can be given here
    for i in range(totalClusters):
        good_points = 0
        bad_points = 0
        for j in range(totalClusters):
            if j != i:
                distance = gpxpy.geo.haversine_distance(latitude_1 = regionCenters[i][0], longitude_1 = regionCenters[i][1], latitude_2 = regionCenters[j][0], longitude_2 = regionCenters[j][1])
                distance = distance/(1.60934*1000)   #distance from meters to miles
                min_distance = min(min_distance, distance) #it will return minimum of "min_distance, distance".
                if distance < 2:
                    good_points += 1
                else:
                    bad_points += 1
        less_dist.append(good_points)
        more_dist.append(bad_points)
    print("On choosing a cluster size of {}".format(totalClusters))
    print("Avg. Number clusters within vicinity where inter cluster distance < 2 miles is {}".format(np.ceil(sum(less_dist)/len(less_dist))))
    print("Avg. Number clusters outside of vicinity where inter cluster distance > 2 miles is {}".format(np.ceil(sum(more_dist)/len(more_dist))))
    print("Minimum distance between any two clusters = {}".format(min_distance))
    print("-"*10)
            
def makingRegions(noOfRegions):
    regions = MiniBatchKMeans(n_clusters = noOfRegions, batch_size = 10000, random_state = 5).fit(coord)
    regionCenters = regions.cluster_centers_ 
    totalClusters = len(regionCenters)
    return regionCenters, totalClusters


# In[25]:


from datetime import datetime

startTime = datetime.now() #saving current timestamp
for i in range(10, 100, 10): #loop from 10-99 withstep of 10 
    regionCenters, totalClusters = makingRegions(i) # calling function making region with value of i from above loop
    min_distance(regionCenters, totalClusters)
print("Time taken = "+str(datetime.now() - startTime))


# In[26]:


df.show()


# In[27]:


df.select("pick_lat", "pick_lng").show()


# In[28]:


coord = np.array(df.select("pick_lat", "pick_lng").collect())


# In[29]:


coord


# In[31]:


import numpy as np
from pyspark.sql.functions import lit 

# Convert DataFrame column to RDD, extract coordinates, and collect as a list
coord = df.select("pick_lat", "pick_lng").rdd.map(lambda row: (row.pick_lat, row.pick_lng)).collect()

# Convert list of coordinates to NumPy array
coord_array = np.array(coord)

# Fit MiniBatchKMeans on the NumPy array
regions = MiniBatchKMeans(n_clusters=50, batch_size=10000, random_state=0).fit(coord_array)

# Predict clusters for each coordinate and convert to a list
cluster_labels = regions.predict(coord_array).tolist()

# Add cluster labels as a new column to the DataFrame
df = df.withColumn("pickup_cluster", lit(cluster_labels))


# In[ ]:


df.show()


# In[ ]:


import matplotlib.pyplot as plt

# Convert Spark DataFrame to Pandas DataFrame
pandas_df = df.toPandas()

bangalore_latitude_range = (12.8340125, 13.1436649)
bangalore_longitude_range = (77.4601025, 77.7840515)
fig = plt.figure()
ax = fig.add_axes([0,0,1.5,1.5])
ax.scatter(
    x=pandas_df.pick_lng.values[:100000].tolist(),
    y=pandas_df.pick_lat.values[:100000].tolist(),
    c=pandas_df.pickup_cluster.values[:100000].tolist(),
    cmap="Paired",
    s=5
)
ax.set_xlim(77.4601025, 77.7840515)
ax.set_ylim(12.8340125, 13.1436649)
ax.set_title("Regions in Bangalore")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


# In[ ]:


# Specify the output path on Azure Blob Storage
output_container_name = "azureml-blobstore-c20a3cf7-1351-4b9e-b08a-e2dfe7d33b91"
output_blob_path = f"wasbs://azureml-blobstore-c20a3cf7-1351-4b9e-b08a-e2dfe7d33b91@ola5132340602.blob.core.windows.net/final_file"

# Drop the `features` column
df_dropped = df.drop("features")

# Coalesce the DataFrame to a single partition
df_coalesced = df_dropped.coalesce(1)

# Write the DataFrame to Azure Blob Storage
df_coalesced.write.format("csv").mode("overwrite").option("header", "true").save(output_blob_path)


# In[ ]:




