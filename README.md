# Boston-mobile-user-delivery-diner-classification
Mobile User Point of interests in Boston Area

This project analyzes mobile users' point of interestin the US and classifies them into different groups diner/delivery worker at Boston area. It is similar to anther project where I look at hotel/non-hotel traveler patterns in the U.S. The user point of interest dataset includes:

0: unix timestamp (0 means 00:00:00 Jan. 1, 1970. Every passing second will add 1),
1: user id,
2: device type (don't need to worry about it)
3: latitude
4: longitude
5: precision of lat and lon. The unit is meters.
6: time difference, add 6 and 1 will give you local time.

To incorporate the POI dataset with the local commerical coordinates(i.e, restaurants, hotels, etc), a US geographical coordinates dataset from Safegraph was collected.
