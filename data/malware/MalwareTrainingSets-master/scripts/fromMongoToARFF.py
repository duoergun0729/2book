# IMPORTANT: This Script is not suitable for production environments !
# IMPORTANT: This Script is just a MOCUKP and it is not performant at all !
# IMPORTANT: I am not accepting criticism on such piece of code since it has been written in Hurry just to make it works
# If you want to know more about what this script does please visit: http://marcoramilli.blogspot.com

# HOW to use it:
# Step 1: imports JSON representation of MIST format into mongodb server. You might want to use a simple bash script such as:
#         for i in **/*.json                                                                                                                                                                                                                                          1 ↵
#         do
#           mongoimport --db test --collection test --file $i
#         done
#
# Step 2: Use this script to convert the JSON representation of MIST file into a ARFF format. Change the variable as you wish.
# Step 3: You are ready to use the ARFF file with your favorite Machine Learning software

from pymongo import MongoClient
import sys

################################################################################
client = MongoClient('localhost', 27017) #Change ME !
db = client['test'] #Change ME !
collection = db['test']#Change ME !
out = open("/tmp/ML.arff", "w")#Change ME!
################################################################################

key_list = []
labels = []

#filling properties
print "[+] Filling UP properties"
total_collections = collection.find().count()
for o, item in enumerate(collection.find(no_cursor_timeout=True)):
    print "|-> Working on Item number: " + str(o) + " on totals: " + str(total_collections)
    for key in item['properties']: 
        if key == "label":
            print "|--> Found Label: " + str(key)
            if item['properties'][key] not in labels:
                print "|--> Append  Label"
                labels.append(item['properties'][key])
        else:
            #testing if is multi properiets
            ps = item['properties'][key].split()
            if len(ps) > 1:
                c=0
                for p in ps:
                    print "|---> Split proerities: " + str(c) + " on total:" + str(len(ps)) + "\r",
                    n_key = str(key) + "!" + str(c)
                    c = c + 1
                    if n_key not in key_list:
                        key_list.append(n_key)
            else:
                if key not in key_list:
                    print "|--> Adding properties: " + str(key)
                    key_list.append(key)
                

#writing header
out.write("@RELATION maware \n")

print "Writing to file header"
for i, k in enumerate(key_list):
    if (i+1) == len(key_list):
        #The last one
        out.write("@ATTRIBUTE '" + k + "' numeric \n")
        out.write("@ATTRIBUTE class {")
        for c, l in enumerate(labels):
            if (c+1) == len(labels):
                out.write("'" + l + "'}\n")
            else:
                out.write("'" + l + "',")
            
    else:
        #No the last one
        out.write("@ATTRIBUTE '" + k + "' numeric \n")
        

def write_data(f, t):
    #writing data
    print "write to file data"
    for o, item in enumerate(collection.find( no_cursor_timeout=True)[f:t]):
        print "|-> Working on Item number: " + str(o) + " on totals: " + str(total_collections)
        for i, key in enumerate(key_list):
            try:
                if key.find("!") != -1:
                    index = key.split('!')[1]
                    property_name = key.split('!')[0]
                    print "**index: " + str(index) + " name: " + str(property_name)
                    value = item['properties'][property_name].split(' ')[int(index)]
                    # interesting ridiculous approach ! :D 
                    value = str( int(value.encode('hex'),16) )
                    print "|---> Value: " + str(value)
                else:
                    value = item['properties'][k]
                    # interesting ridicoulous approach ! :D 
                    value = str( int(value.encode('hex'),16) )
            except Exception as e:
                print "Exception: " + str(e)
                value = "?"
            if (i+1) == len(key_list):
                out.write(value + "," +  item['properties']['label'] + "\n")
            else:
                out.write(value + ",")
    

out.write("@DATA \n")
f = 0
t = 0
step = 2

#Just to fill down memory .....
while (t <= total_collections -1):
    t = t + step
    write_data(f,t)
    f = t + 1

out.close()
print "[*] I am done !"
