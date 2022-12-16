This example is constructed:

```
Triple: /m/0dg3n1  /base/locations/continents/countries_within  /m/01699
Triple translated: Africa  /base/locations/continents/countries_within  Burkina Faso
	Head Predictions:
	    /m/0ncy4 # Wimbledon (suburb of London) -> random wrong
	    /m/0dg3n1 # Africa (continent on the Earth's northern and southern hemispheres) -> test correct
	Tail Predictions:
		/m/0dbks # Douala   (city in Cameroon) -> wrong
		/m/027jk # Djibouti   (country in Africa) -> test
		/m/02khs # Eritrea   (country in the Horn of Africa) -> train
		/m/088q4 # Zimbabwe   (country in Africa) -> valid
		/m/01699 # Burkina Faso   (country in Africa) -> correct
```