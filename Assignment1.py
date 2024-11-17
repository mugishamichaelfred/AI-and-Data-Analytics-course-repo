""" Consider the lists countries and cities
"""
countries=["Rwanda","DRC","CAR","Gabon","Congo"]
cities=["Kigali","Kinshasa","Bangui","Libreville","Brazzaville"]
""" Use a loop to create a list of tuples, where each tuple 
represents a country and  its capital city """
###your code starts here


country_city_pairs = []
for i in range(len(countries)):
    country_city_pairs.append((countries[i], cities[i]))

print(country_city_pairs)



### your code ends here

""" Create the same list of tuples but using the zip() function
"""

###Your code start here


countries_cities_pairs = []
for country, city in zip(countries, cities):
    countries_cities_pairs.append((country, city))

print(countries_cities_pairs)


##your code ends here

""" Change the list of tuples you have created above into 
a list of lists
"""
###your code starts here


# List of tuples
country_city_pairs = [
    ('Rwanda', 'Kigali'),
    ('DRC', 'Kinshasa'),
    ('CAR', 'Bangui'),
    ('Gabon', 'Libreville'),
    ('Congo', 'Brazzaville')
]

# Convert to a list of lists
country_city_lists = [list(pair) for pair in country_city_pairs]

print(country_city_lists)



###your code ends here

""" The following lists 
"""
countries_1=["Germany","France","Belgium","India"]*2
cities_1=["Frankfurt","Paris","Liege","Bangalore","Bonn","Bordeaux","Brussels","Dehli"]

""" Use a loop to create a dictionary where, the key is
 a country and the value is the list of cities in that country"""

### Your code starts here


country_city_dict = {}
for i in range(len(countries_1)):
    country = countries_1[i]
    city = cities_1[i]
    if country not in country_city_dict:
        country_city_dict[country] = []
    country_city_dict[country].append(city)
print(country_city_dict)


###your code ends here

""" Do the same task as above , but this time use dictionary comprehension
"""
### Your code starts here


country_city_dict = {
    country: [cities_1[i] for i in range(len(countries_1)) if countries_1[i] == country]
    for country in set(countries_1)
}

print(country_city_dict)


####Your code ends here


