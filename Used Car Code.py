import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
# Loading Data
start_time = time.time()
training_data = pd.read_csv("Training_DataSet.csv")
#print(training_data)
test_data = pd.read_csv("Test_DataSet.csv")
# Finding Null Values
#print(training_data.VehPriceLabel.isna().sum())
# Filling null values with 'None'
training_data['VehPriceLabel'] = training_data.VehPriceLabel.fillna('None')
# Checking that the label was added
#print(training_data.VehPriceLabel.value_counts())
# Creating Dictionary to map for Vehicle Price Label
VehPriceLabel = {'None':0,
                 'Fair Price': 1,
                 'Good Deal': 2,
                 'Great Deal': 3}
# Mapping dictionary to data for ML
training_data['VehPriceLabel'] = training_data.VehPriceLabel.map(VehPriceLabel)
# Confirming it worked
#print(training_data.VehPriceLabel)
#print(training_data.describe())
#print(training_data.VehType.dtype)
# Dropping values that won't be used in my ML model
columns_to_drop = ['VehBodystyle',  'VehEngine', 'VehHistory', 'SellerCity', 'SellerState', 'SellerName', 'VehFeats', 'SellerZip', 'VehSellerNotes', 'VehType']
training_data.drop(columns= columns_to_drop, inplace=True)
# Iterating through columns looking for inconsistancies
#for column in training_data.columns:
    #if training_data[column].dtype == 'object':
        #print(training_data[column].value_counts())
        #print('\n')
# Correcting inconsistencies
training_data.VehDriveTrain.replace({'Four Wheel Drive':'4WD', 
                                     'FRONT-WHEEL DRIVE':'FWD', 
                                     'ALL-WHEEL DRIVE':'AWD', 
                                     'Front Wheel Drive':'FWD', 
                                     'All Wheel Drive':'AWD',
                                     '4x4':'4X4',
                                     '4x4/4-wheel drive':'4x4/4WD',
                                     'All-wheel Drive':'AWD',
                                     'Front-wheel Drive':'FWD',
                                     'ALL-WHEEL DRIVE WITH LOCKING AND LIMITED-SLIP DIFFERENTIAL':'AWD',
                                     '2WD':'FWD',
                                     'ALL WHEEL':'AWD',
                                     'AllWheelDrive':'AWD'}, inplace=True)
training_data.VehTransmission.replace({'Automatic':'8-Speed Automatic',
                                       'AUTOMATIC':'8-Speed Automatic',
                                       'Automatic 8-Speed':'8-Speed Automatic',
                                       '8-SPEED A/T': '8-Speed Automatic',
                                       '8-Speed A/T': '8-Speed Automatic',
                                       '8-Speed':'8-Speed Automatic',
                                       'Automatic, 8-Spd':'8-Speed Automatic',
                                       '8 Speed Automatic': '8-Speed Automatic',
                                       '8-SPEED AUTOMATIC':'8-Speed Automatic',
                                       'A':'8-Speed Automatic',
                                       '8 speed automatic':'8-Speed Automatic',
                                       'Shiftable Automatic':'8-Speed Shiftable Automatic',
                                       '8-Spd Auto 850RE Trans (Make)':'8-Speed Automatic',
                                       'aujtomatic':'8-Speed Automatic',
                                       'Auto':'8-Speed Automatic',
                                       '8-Spd Auto 850RE Trans (Make':'8-Speed Automatic',
                                       'MRC':'8-Speed Automatic',
                                       'automatic':'8-Speed Automatic',
                                       'AUTO':'8-Speed Automatic',
                                       'a':'8-Speed Automatic',
                                       'Automatic w/OD':'8-Speed Automatic w/OD',
                                       '8-Speed TorqueFlite Automatic':'8-Spd TorqueFlite Automatic',
                                       'Automanual':'8-Speed Shiftable Automatic'}, inplace=True)
training_data.Vehicle_Trim.replace({'Limited 75th Anniversary Edition':'75th Anniversary',
                                    '75th Anniversary Edition':'75th Anniversary',
                                    'Limited 4x4':'Limited',
                                    'Limited 75th Anniversary':'75th Anniversary',
                                    'Limited X':'Limited',
                                    'Luxury FWD':'Luxury',
                                    'Premium Luxury FWD':'Premium Luxury',
                                    'Platinum AWD':'Platinum',
                                    'FWD':'Base',
                                    'Luxury AWD':'Luxury',
                                    'Premium Luxury AWD':'Premium Luxury',
                                    'SRT Night':'SRT'}, inplace=True)
# Looking at just Exterior and Interior colors due to size of the value counts
#pd.set_option("display.max_rows", None)
#print(training_data.VehColorExt.value_counts())
training_data.VehColorExt.replace({'Crystal White Tri-Coat':'Crystal White Tricoat',
                                   'Billet Silver Metallic Clearcoat - Silver':'Billet Silver Metallic Clearcoat',
                                   'Grey':'Gray',
                                   'Billett Silver Clearcoat Metallic':'Billet Silver Metallic Clearcoat',
                                   'Billet Silver Metallic Clear Coat': 'Billet Silver Metallic Clearcoat',
                                   'Unspecified':'Not Specified',
                                   'Black Black':'Black',
                                   'Red Passion Tc':'Red Passion Tintcoat',
                                   'Dk. Gray':'Dark Gray',
                                   'Granite Crystal Metallic Clearcoat - Gray':'Granite Crystal Metallic Clearcoat',
                                   'Crystal White Tricoa':'Crystal White Tricoat',
                                   'Cashmere Pearl Coat':'Cashmere Pearlcoat',
                                   'Granite Chrystal Metallic Clearcoat':'Granite Crystal Metallic Clearcoat',
                                   'Brilliant Black Crystal Pearl Coat':'Brilliant Black Crystal Pearlcoat',
                                   'Brilliant Black Crystal Pearl':'Brilliant Black Crystal Pearl',
                                   'Brilliant Black Crystal P':'Brillian Black Crystal Pearlcoat',
                                   'Granite Crystal Clearcoat Metallic':'Granite Crystal Metallic Clearcoat',
                                   'Ivory 3 Coat': 'Ivory 3-Coat',
                                   'Bright White Clear Coat':'Bright White Clearcoat',
                                   'Granite Crystal Metallic Clear Coat':'Granite Cystal Metallic Clearcoat',
                                   'Diamond Black Crystal Pearl-coat':'Diamond Black Crystal Pearlcoat',
                                   'Diamond Black Crystal Pearl Coat':'Diamond Black Crystal Pearlcoat',
                                   'Diamond Black Crystal P/C':'Diamond Black Crystal Pearlcoat',
                                   'Diamond Black Crystal':'Diamond Black Crystal Pearlcoat',
                                   'Harbor Blue Met':'Harbor Blue Metallic',
                                   'G35 Harbor Blue Metallic':'Harbor Blue Metallic',
                                   'Maximum Steel Clearcoat Metallic':'Maximum Steel Metallic Clearcoat',
                                   'Maximum Steel Metallic Clear Coat': 'Maximum Steel Metallic Clearcoat',
                                   'Maximum Steel Met. Clear Coat':'Maximum Steel Metallic Clearcoat',
                                   'Maximum Steel M':'Maximum Steel Metallic',
                                   'Billet Silver M': 'Billet Silver Metallic',
                                   'Velvet Red Pearl Coat':'Velvet Red Pearlcoat',
                                   'Certified Lthr Roof Nav Camera':'Not Specified',
                                   'Certified Lthr Pano Roof Hot Cold Seats':'Not Specified',
                                   'Certified Roof Camera Htd Seats':'Not Specified',
                                   'Silver Coast Me':'Silver Coast Metallic',
                                   'Beigh':'Beige',
                                   'Sangria Metallic Clear Coat':'Sangria Metallic Clearcoat',
                                   'Billet Silver M':'Billet Silver Metallic',
                                   'Certified Hemi Lthr Pano Roof Nav':'Not Specified',
                                   'Certified Camera Bluetooth Alloys':'Not Specified',
                                   'Certified Lthr Pano Roof Nav Camera':'Not Specified',
                                   'Silver Awd Pano Roof Loaded':'Not Specified',
                                   'Black Forest Green Pearl Coat':'Black Forest Green Pearlcoat',
                                   'Stellar Black M':'Stellar Black Metallic',
                                   'Billiet Silver':'Billet Silver',
                                   'Deep Cherry Red Chrystal Pearl Coat':'Deep Cherry Red Crystal Pearlcoat',
                                   'True Blue Pearl Coat':'True Blue Pearlcoat',
                                   'True Blue Pearl':'True Blue Pearlcoat',
                                   'Velvet':'Velvet Red',
                                   'Redline 2 Coat Pearl **Night Edition**':'Redline 2 Coat Pearl',
                                   'Redline Pearl Coat':'Redline 2 Coat Pearl',
                                   'Granite Crystal Met. Clear Coat':'Granite Crystal Metallic Clearcoat',
                                   'Db Black':'Black',
                                   'Pewter':'Other',
                                   'Granite Crystal Gray':'Other',
                                   'Bright Sil':'Other',
                                   'Granite Metallic':'Other',
                                   'Shadow Metallic':'Other',
                                   'Summit White':'Other',
                                   'Midnight Sky Metallic':'Other',
                                   'Dark Red':'Other',
                                   'Black Crystal':'Other',
                                   'White':'Other',
                                   'Billet':'Other',
                                   'Ruby Red Metallic':'Other',
                                   'Harbor Blue':'Other',
                                   'Deep Amethyst':'Other',
                                   'Pearl':'Other',
                                   'Pearl White':'Other',
                                   'Silver Coast':'Other',
                                   'Mineral Gray':'Other',
                                   'Black Limited':'Other',
                                   'Silver Metallic':'Other',
                                   'Brilliant Black Crystal Clear':'Maximum Steel',
                                   'Maximum Steel':'Other',
                                   'Silver Black':'Other',
                                   'Dark Brown':'Other',
                                   'Platinum':'Other',
                                   'Steel Gray':'Other',
                                   'Gy':'Gray',
                                   'Brilliant Black Crystal Pearl':'Brilliant Black Crystal Pearlcoat',
                                   'Brillian Black Crystal Pearlcoat':'Brilliant Black Crystal Pearlcoat',
                                   'Diamond Black Crystal Pearl-coat':'Diamond Black Crystal Pearlcoat',
                                   'Billet Silver Clearcoat Metallic':'Billet Silver Metallic Clearcoat',
                                   'Maximum Steel Metallic Cc':'Maximum Steel Metallic Clearcoat',
                                   'Dark Granite M':'Dark Granine Metallic',
                                   'Deep Cherry Red Crystal Pearl Coat':'Deep Cherry Red Crystal Pearlcoat',
                                   'Undetermined':'Other',
                                   'Gb8 Stellar Black Metallic':'Stellar Black Metallic',
                                   'Not Specified':'Other',
                                   'Granite Met':'Granite Metallic'}, inplace=True)
training_data.VehColorInt.replace({'Shara Beige':'Sahara Beige',
                                   'black':'Black',
                                   'Light Frost / Brown':'Light Frost/Brown',
                                   'jet black':'Jet Black',
                                   'Light Frost / Black':'Light Frost/Black',
                                   'BLACK':'Black',
                                   'Black / Dark Sienna Brown':'Black/Dark Sienna Brown',
                                   'sahara beige':'Sahara Beige',
                                   'Sepia / Black':'Sepia/Black',
                                   'lt frost beige black':'Light Frost Beige/Black',
                                   'Lt Frost Beige/Black':'Light Frost Beige/Black',
                                   'JET BLACK LUNAR BRUSHED ALUMINIUM TRIM':'Jet Black',
                                   'Indigo Blue / Brown':'Brown/Indigo Blue',
                                   'Black/Light Frost Beige':'Light Frost Beige/Black',
                                   'CIRRUS W/ DARK TITANIUM ACCENTS DIAMOND CUT ALUMIN':'Cirrus',
                                   'SAHARA BEIGE/JET BLACK ACCENTS NATURAL SAPELE HIGH':'Sahara Beige/Jet Black',
                                   'Sahara Beige W/ Jet Black Accent':'Sahara Beige/Jet Black',
                                   'Jet Black w/Full Leather Seats w/Mini Perforated I':'Jet Black Leather',
                                   'SAHARA BEIGE/JET BLACK ACCENTS OKAPI STRIPE DESIGN': 'Sahara Beige/Jet Black',
                                   'Ruby Red/black':'Ruby Red/Black',
                                   'BLACK LEATHER':'Black Leather',
                                   'JET BLACK':'Jet Black',
                                   'Jet Black Lunar Brushed Alumninium Trim':'Jet Black',
                                   'Jet Black, premium leather':'Jet Black Leather',
                                   'Black Cloth':'Black',
                                   'Black, cloth':'Cloth',
                                   'Beige Cloth':'Beige',
                                   'Jet Black w/Leather Seating Surfaces w/Mini Perfor':'Jet Black Leather',
                                   'Cirrus w/Dark Titanium Accents w/Full Leather Seat':'Cirrus',
                                   'Black/Lt. Frost Beige':'Light Frost Beige/Black',
                                   'GREY,LEATHER':'Gray',
                                   'Light Grey/Black':'Light Gray/Black',
                                   'Ruby Red / Black':'Ruby Red/Black',
                                   'CIRRUS':'Cirrus',
                                   'SIENNA BROWN':'Sienna Brown',
                                   'light frost brown':'Light Frost Brown',
                                   'Jet Black W/Leather Seating Surfaces W/Mini Perfor':'Jet Black Leather',
                                   'JET BLACK, LEATHER SEATING SURFACES WITH MINI-PERF':'Jet Black Leather',
                                   'TAN':'Tan',
                                   'BEIGE':'Beige',
                                   'brown':'Brown',
                                   'Black / Light Frost Beige':'Light Frost Beige/Black',
                                   'gray':'Gray',
                                   'Cirrus w/Dark Titanium Accents w/Leather Seating S':'Cirrus',
                                   'Black/Ruby Red':'Ruby Red/Black',
                                   'CHARCOAL':'Charcoal',
                                   'Light Gray / Black':'Light Gray/Black',
                                   'JET BLACK BRONZE CARBON FIBER TRIM':'Jet Black',
                                   'Black / Ruby Red':'Ruby Red/Black',
                                   'Cirrus w/dark atmosphere':'Cirrus',
                                   'SUGAR MAPLE LEATHER':'Maple Sugar',
                                   'Ruby Red/Black Leather':'Ruby Red/Black',
                                   'Not Specified':'Other',
                                   'Light Frost Beige Black':'Light Frost Beige/Black',
                                   'Tan Leather':'Tan',
                                   'Gray - Black':'Light Gray/Black',
                                   'Sahara Beige w/Jet Black Accents w/Full Leather Se':'Sahara Beige w/ Jet Black Accent',
                                   'Light Frost Brown':'Light Frost/Brown',
                                   'Ruby Red/Black Leather/Suede':'Ruby Red/Black',
                                   'Carbon':'Carbon Plum',
                                   'Sahara Beige Leather':'Sahara Beige',
                                   'Jet Black Leather':'Jet Black',
                                   'Beige Leather':'Beige',
                                   'Brown Leather':'Brown',
                                   'Red / Black':'Ruby Red/Black',
                                   'Dark Ruby Red/Black':'Ruby Red/Black',
                                   'Red/Black':'Ruby Red/Black',
                                   'Red':'Ruby Red',
                                   'Pewter':'Other',
                                   'BLACK/LIGHT FROST':'Other',
                                   'Graphite':'Other',
                                   'Dark Granite':'Other',
                                   'Light Frost Beige / Black':'Light Frost Beige/Black',
                                   'Jet Black Lunar Brushed Aluminium Trim':'Other',
                                   'Bronze':'Other',
                                   'Ebony':'Other',
                                   'Cloth':'Other'}, inplace=True)
#print(training_data.VehColorInt.value_counts())
# Validating Replacements
for column in training_data.columns:
    if training_data[column].dtype == 'object':
        print(training_data[column].value_counts())
        print('\n')
# Creating a function to deal with null values in other categories and using Label Encoder to convert to numerical values
def unknown_and_encoder(dataframe):
    label_encoder = LabelEncoder()
    dataframe.SellerListSrc = dataframe.SellerListSrc.fillna('Other')
    dataframe.SellerListSrc = label_encoder.fit_transform(dataframe.SellerListSrc)
    dataframe.VehColorExt = dataframe.VehColorExt.fillna('Other')
    dataframe.VehColorExt = label_encoder.fit_transform(dataframe.VehColorExt)
    dataframe.VehColorInt = dataframe.VehColorInt.fillna('Other')
    dataframe.VehColorInt = label_encoder.fit_transform(dataframe.VehColorInt)
    dataframe.VehDriveTrain = dataframe.VehDriveTrain.fillna('Unknown')
    dataframe.VehDriveTrain = label_encoder.fit_transform(dataframe.VehDriveTrain)
    #dataframe.VehEngine = dataframe.VehEngine.fillna('Unknown')
    #dataframe.VehEngine = label_encoder.fit_transform(dataframe.VehEngine)
    dataframe.VehFuel = dataframe.VehFuel.fillna('Gasoline')
    #dataframe.VehHistory = dataframe.VehHistory.fillna('Unknown')
    #dataframe.VehHistory = label_encoder.fit_transform(dataframe.VehHistory)
    dataframe.VehTransmission = dataframe.VehTransmission.fillna('Not Specified')
    dataframe.VehTransmission = label_encoder.fit_transform(dataframe.VehTransmission)
    # For numerical values, filling null values with mean
    dataframe.VehListdays = dataframe.VehListdays.fillna(dataframe.VehListdays.mean())
    dataframe.VehMileage = dataframe.VehMileage.fillna(dataframe.VehMileage.mean())
    # Converting Boolean to binary
    dataframe.SellerIsPriv.replace({False:0, True:1}, inplace=True)
    dataframe.VehCertified.replace({False:0, True:1}, inplace=True)
    return dataframe
training_data = unknown_and_encoder(training_data)
training_data.replace({'Unknown':'Gasoline'}, inplace=True)
#print(training_data) 
# Dropping any null values left
training_data.dropna(inplace=True)
#print(training_data.isna().sum())
#print(training_data)
# Convert categorical data to dummies
#print("Columns before Dummying:", training_data.columns)
columns_to_dummy = ['VehMake', 'VehFuel', 'VehModel']
training_data = pd.get_dummies(training_data, columns=columns_to_dummy)
#print(training_data)
# Creating X values and y value
y_reg = training_data.Dealer_Listing_Price
y_classification = training_data.Vehicle_Trim
X = training_data.drop(columns=['Dealer_Listing_Price', 'Vehicle_Trim'])
#print("Shape of X:", X.shape)
#print(y_reg)
#print(y_classification)
# Creating training data set and testing data set
X_train, X_val, y_train, y_val = train_test_split(X, y_reg, test_size=.1, random_state=123)
X_train_class, X_val_class, y_train_class, y_val_class = train_test_split(X, y_classification, test_size=.2, random_state=123)
# Creating parameter grid to optimize Random Forest 
n_estimators = [int(x) for x in np.linspace(80, 100, 3)]
max_features = ['auto', 'sqrt', 'log2', None]
max_depth = [int(x) for x in np.linspace(70, 100, 4)]
min_samples_split = [int(x) for x in np.linspace(2, 10, 9)]
min_samples_leaf = [1, 2, 3, 4]
bootstrap = [True, False]
param_grid_random = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# Initiating Random Forest Regressor
rf_reg = RandomForestRegressor(random_state=123)
# Using GridSearchCV to find best parameters
#rf_reg_grid = RandomizedSearchCV(estimator=rf_reg, param_distributions=param_grid_random, cv=3, n_iter=200, n_jobs= -1)
#rf_reg_grid.fit(X_val, y_val)
#print("Best Parameters Regression:", rf_reg_grid.best_params_)
rf_reg.fit(X, y_reg)
# Score on training data
#print("Regression Training Score:", rf_reg.score(X_train, y_train))
# Score on training data
#print("Regression Validation Score", rf_reg.score(X_val, y_val))
# Using Random Forest Classifer
rf_class = RandomForestClassifier(random_state=32, n_estimators=80, bootstrap=False, criterion='entropy', min_samples_split=7, min_samples_leaf=1)
# Using GridSearchCV to find best parameters
#rf_class_grid = RandomizedSearchCV(estimator=rf_class, param_distributions=param_grid_random, n_iter=400, cv= 3, n_jobs= 25)
#rf_class_grid.fit(X_val_class, y_val_class)
#print(rf_class_grid.best_params_)
#print(rf_class_grid.best_score_)
rf_class.fit(X, y_classification)
#print("Classification Training Score:", rf_class.score(X_train_class, y_train_class))
#print("Classification Validation Score:", rf_class.score(X_val_class, y_val_class))
# Working on test data
# Dropping columns that won't be used for ML
test_data.drop(columns=columns_to_drop, inplace=True)
test_data['VehPriceLabel'] = test_data.VehPriceLabel.fillna('None')
VehPriceLabel = {'None':0,
                 'Fair Price': 1,
                 'Good Deal': 2,
                 'Great Deal': 3}
# Mapping dictionary to data for ML
test_data['VehPriceLabel'] = test_data.VehPriceLabel.map(VehPriceLabel)
test_data.VehColorExt.replace({'Crystal White Tri-Coat':'Crystal White Tricoat',
                                   'Billet Silver Metallic Clearcoat - Silver':'Billet Silver Metallic Clearcoat',
                                   'Grey':'Gray',
                                   'Billett Silver Clearcoat Metallic':'Billet Silver Metallic Clearcoat',
                                   'Billet Silver Metallic Clear Coat': 'Billet Silver Metallic Clearcoat',
                                   'Unspecified':'Not Specified',
                                   'Black Black':'Black',
                                   'Red Passion Tc':'Red Passion Tintcoat',
                                   'Dk. Gray':'Dark Gray',
                                   'Granite Crystal Metallic Clearcoat - Gray':'Granite Crystal Metallic Clearcoat',
                                   'Crystal White Tricoa':'Crystal White Tricoat',
                                   'Cashmere Pearl Coat':'Cashmere Pearlcoat',
                                   'Granite Chrystal Metallic Clearcoat':'Granite Crystal Metallic Clearcoat',
                                   'Brilliant Black Crystal Pearl Coat':'Brilliant Black Crystal Pearlcoat',
                                   'Brilliant Black Crystal Pearl':'Brilliant Black Crystal Pearl',
                                   'Brilliant Black Crystal P':'Brillian Black Crystal Pearlcoat',
                                   'Granite Crystal Clearcoat Metallic':'Granite Crystal Metallic Clearcoat',
                                   'Ivory 3 Coat': 'Ivory 3-Coat',
                                   'Bright White Clear Coat':'Bright White Clearcoat',
                                   'Granite Crystal Metallic Clear Coat':'Granite Cystal Metallic Clearcoat',
                                   'Diamond Black Crystal Pearl-coat':'Diamond Black Crystal Pearlcoat',
                                   'Diamond Black Crystal Pearl Coat':'Diamond Black Crystal Pearlcoat',
                                   'Diamond Black Crystal P/C':'Diamond Black Crystal Pearlcoat',
                                   'Diamond Black Crystal':'Diamond Black Crystal Pearlcoat',
                                   'Harbor Blue Met':'Harbor Blue Metallic',
                                   'G35 Harbor Blue Metallic':'Harbor Blue Metallic',
                                   'Maximum Steel Clearcoat Metallic':'Maximum Steel Metallic Clearcoat',
                                   'Maximum Steel Metallic Clear Coat': 'Maximum Steel Metallic Clearcoat',
                                   'Maximum Steel Met. Clear Coat':'Maximum Steel Metallic Clearcoat',
                                   'Maximum Steel M':'Maximum Steel Metallic',
                                   'Billet Silver M': 'Billet Silver Metallic',
                                   'Velvet Red Pearl Coat':'Velvet Red Pearlcoat',
                                   'Certified Lthr Roof Nav Camera':'Not Specified',
                                   'Certified Lthr Pano Roof Hot Cold Seats':'Not Specified',
                                   'Certified Roof Camera Htd Seats':'Not Specified',
                                   'Silver Coast Me':'Silver Coast Metallic',
                                   'Beigh':'Beige',
                                   'Sangria Metallic Clear Coat':'Sangria Metallic Clearcoat',
                                   'Billet Silver M':'Billet Silver Metallic',
                                   'Certified Hemi Lthr Pano Roof Nav':'Not Specified',
                                   'Certified Camera Bluetooth Alloys':'Not Specified',
                                   'Certified Lthr Pano Roof Nav Camera':'Not Specified',
                                   'Silver Awd Pano Roof Loaded':'Not Specified',
                                   'Black Forest Green Pearl Coat':'Black Forest Green Pearlcoat',
                                   'Stellar Black M':'Stellar Black Metallic',
                                   'Billiet Silver':'Billet Silver',
                                   'Deep Cherry Red Chrystal Pearl Coat':'Deep Cherry Red Crystal Pearlcoat',
                                   'True Blue Pearl Coat':'True Blue Pearlcoat',
                                   'True Blue Pearl':'True Blue Pearlcoat',
                                   'Velvet':'Velvet Red',
                                   'Redline 2 Coat Pearl **Night Edition**':'Redline 2 Coat Pearl',
                                   'Redline Pearl Coat':'Redline 2 Coat Pearl',
                                   'Granite Crystal Met. Clear Coat':'Granite Crystal Metallic Clearcoat',
                                   'Db Black':'Black',
                                   'Pewter':'Other',
                                   'Granite Crystal Gray':'Other',
                                   'Bright Sil':'Other',
                                   'Granite Metallic':'Other',
                                   'Shadow Metallic':'Other',
                                   'Summit White':'Other',
                                   'Midnight Sky Metallic':'Other',
                                   'Dark Red':'Other',
                                   'Black Crystal':'Other',
                                   'White':'Other',
                                   'Billet':'Other',
                                   'Ruby Red Metallic':'Other',
                                   'Harbor Blue':'Other',
                                   'Deep Amethyst':'Other',
                                   'Pearl':'Other',
                                   'Pearl White':'Other',
                                   'Silver Coast':'Other',
                                   'Mineral Gray':'Other',
                                   'Black Limited':'Other',
                                   'Silver Metallic':'Other',
                                   'Brilliant Black Crystal Clear':'Maximum Steel',
                                   'Maximum Steel':'Other',
                                   'Silver Black':'Other',
                                   'Dark Brown':'Other',
                                   'Platinum':'Other',
                                   'Steel Gray':'Other',
                                   'Gy':'Gray',
                                   'Brilliant Black Crystal Pearl':'Brilliant Black Crystal Pearlcoat',
                                   'Brillian Black Crystal Pearlcoat':'Brilliant Black Crystal Pearlcoat',
                                   'Diamond Black Crystal Pearl-coat':'Diamond Black Crystal Pearlcoat',
                                   'Gb8 Stellar Black Metallic':'Stellar Black Metallic',
                                   'Billet Silver Clearcoat Metallic':'Billet Silver Metallic Clearcoat',
                                   'Maximum Steel Metallic Cc':'Maximum Steel Metallic Clearcoat',
                                   'Deep Cherry Red Crystal Pearl Coat':'Deep Cherry Red Crystal Pearlcoat',
                                   'Dark Granite M':'Dark Granite Metallic',
                                   'Granite Met':'Dark Granite Metallic',
                                   'Redline':'Red Line'}, inplace=True)
test_data.VehColorInt.replace({'Shara Beige':'Sahara Beige',
                                   'black':'Black',
                                   'Light Frost / Brown':'Light Frost/Brown',
                                   'jet black':'Jet Black',
                                   'Light Frost / Black':'Light Frost/Black',
                                   'BLACK':'Black',
                                   'Black / Dark Sienna Brown':'Black/Dark Sienna Brown',
                                   'sahara beige':'Sahara Beige',
                                   'Sepia / Black':'Sepia/Black',
                                   'lt frost beige black':'Light Frost Beige/Black',
                                   'Lt Frost Beige/Black':'Light Frost Beige/Black',
                                   'JET BLACK LUNAR BRUSHED ALUMINIUM TRIM':'Jet Black',
                                   'Indigo Blue / Brown':'Brown/Indigo Blue',
                                   'Black/Light Frost Beige':'Light Frost Beige/Black',
                                   'CIRRUS W/ DARK TITANIUM ACCENTS DIAMOND CUT ALUMIN':'Cirrus',
                                   'SAHARA BEIGE/JET BLACK ACCENTS NATURAL SAPELE HIGH':'Sahara Beige/Jet Black',
                                   'Sahara Beige W/ Jet Black Accent':'Sahara Beige/Jet Black',
                                   'Jet Black w/Full Leather Seats w/Mini Perforated I':'Jet Black',
                                   'SAHARA BEIGE/JET BLACK ACCENTS OKAPI STRIPE DESIGN': 'Sahara Beige/Jet Black',
                                   'Ruby Red/black':'Ruby Red/Black',
                                   'BLACK LEATHER':'Black Leather',
                                   'JET BLACK':'Jet Black',
                                   'Jet Black Lunar Brushed Alumninium Trim':'Jet Black',
                                   'Jet Black, premium leather':'Jet Black',
                                   'Black Cloth':'Black',
                                   'Black, cloth':'Cloth',
                                   'Beige Cloth':'Beige',
                                   'Jet Black w/Leather Seating Surfaces w/Mini Perfor':'Jet Black',
                                   'Cirrus w/Dark Titanium Accents w/Full Leather Seat':'Cirrus',
                                   'Black/Lt. Frost Beige':'Light Frost Beige/Black',
                                   'GREY,LEATHER':'Gray',
                                   'Light Grey/Black':'Light Gray/Black',
                                   'Ruby Red / Black':'Ruby Red/Black',
                                   'CIRRUS':'Cirrus',
                                   'SIENNA BROWN':'Sienna Brown',
                                   'light frost brown':'Light Frost Brown',
                                   'Jet Black W/Leather Seating Surfaces W/Mini Perfor':'Jet Black Leather',
                                   'JET BLACK, LEATHER SEATING SURFACES WITH MINI-PERF':'Jet Black Leather',
                                   'TAN':'Tan',
                                   'BEIGE':'Beige',
                                   'brown':'Brown',
                                   'Black / Light Frost Beige':'Light Frost Beige/Black',
                                   'gray':'Gray',
                                   'Cirrus w/Dark Titanium Accents w/Leather Seating S':'Cirrus',
                                   'Black/Ruby Red':'Ruby Red/Black',
                                   'CHARCOAL':'Charcoal',
                                   'Light Gray / Black':'Light Gray/Black',
                                   'JET BLACK BRONZE CARBON FIBER TRIM':'Jet Black',
                                   'Black / Ruby Red':'Ruby Red/Black',
                                   'Cirrus w/dark atmosphere':'Cirrus',
                                   'SUGAR MAPLE LEATHER':'Maple Sugar',
                                   'Ruby Red/Black Leather':'Ruby Red/Black',
                                   'Not Specified':'Other',
                                   'Light Frost Beige Black':'Light Frost Beige/Black',
                                   'Tan Leather':'Tan',
                                   'Gray - Black':'Light Gray/Black',
                                   'Sahara Beige w/Jet Black Accents w/Full Leather Se':'Sahara Beige w/ Jet Black Accent',
                                   'Light Frost Brown':'Light Frost/Brown',
                                   'Ruby Red/Black Leather/Suede':'Ruby Red/Black',
                                   'Carbon':'Carbon Plum',
                                   'Sahara Beige Leather':'Sahara Beige',
                                   'Jet Black Leather':'Jet Black',
                                   'Beige Leather':'Beige',
                                   'Brown Leather':'Brown',
                                   'Red / Black':'Ruby Red/Black',
                                   'Dark Ruby Red/Black':'Ruby Red/Black',
                                   'Red/Black':'Ruby Red/Black',
                                   'Red':'Ruby Red',
                                   'Pewter':'Other',
                                   'BLACK/LIGHT FROST':'Other',
                                   'Graphite':'Other',
                                   'Dark Granite':'Other',
                                   'Light Frost Beige / Black':'Light Frost Beige/Black',
                                   'Jet Black Lunar Brushed Aluminium Trim.*':'Other',
                                   'Light Frost Brown.*':'Other',
                                   'Bronze':'Other',
                                   'Ebony':'Other',
                                   'Cloth':'Other',
                                   'Black Heated Leather':'Black Leather',
                                   'Sahara Beige W/jet Black Accents W/leatherette Sea':'Sahara Beige/Jet Black',
                                   'TAN,LEATHER':'Tan',
                                   'Brown/Light Frost':'Light Frost/Brown',
                                   'RED':'Ruby Red',
                                   'Dark Ruby Red':'Ruby Red',
                                   'LUX2':'Not Specified',
                                   'NOTE: MUST HAVE CUSTOMER SIGN':'Not Specified',
                                   'Black w/Leather Trimmed Bucket Seats or Leather Tr':'Black'}, inplace=True)
test_data.VehDriveTrain.replace({'Four Wheel Drive':'4WD', 
                                     'FRONT-WHEEL DRIVE':'FWD', 
                                     'ALL-WHEEL DRIVE':'AWD', 
                                     'Front Wheel Drive':'FWD', 
                                     'All Wheel Drive':'AWD',
                                     '4x4':'4X4',
                                     '4x4/4-wheel drive':'4x4/4WD',
                                     'All-wheel Drive':'AWD',
                                     'Front-wheel Drive':'FWD',
                                     'ALL-WHEEL DRIVE WITH LOCKING AND LIMITED-SLIP DIFFERENTIAL':'AWD',
                                     '2WD':'FWD',
                                     'ALL WHEEL':'AWD',
                                     'AllWheelDrive':'AWD'}, inplace=True)
test_data.VehTransmission.replace({'Automatic':'8-Speed Automatic',
                                       'AUTOMATIC':'8-Speed Automatic',
                                       'Automatic 8-Speed':'8-Speed Automatic',
                                       '8-SPEED A/T': '8-Speed Automatic',
                                       '8-Speed A/T': '8-Speed Automatic',
                                       '8-Speed':'8-Speed Automatic',
                                       'Automatic, 8-Spd':'8-Speed Automatic',
                                       '8 Speed Automatic': '8-Speed Automatic',
                                       '8-SPEED AUTOMATIC':'8-Speed Automatic',
                                       'A':'8-Speed Automatic',
                                       '8 speed automatic':'8-Speed Automatic',
                                       'Shiftable Automatic':'8-Speed Shiftable Automatic',
                                       '8-Spd Auto 850RE Trans (Make)':'8-Speed Automatic',
                                       'aujtomatic':'8-Speed Automatic',
                                       'Auto':'8-Speed Automatic',
                                       '8-Spd Auto 850RE Trans (Make':'8-Speed Automatic',
                                       'MRC':'8-Speed Automatic',
                                       'automatic':'8-Speed Automatic',
                                       'AUTO':'8-Speed Automatic',
                                       'a':'8-Speed Automatic',
                                       'Automatic w/OD':'8-Speed Automatic w/OD',
                                       '8-Speed TorqueFlite Automatic':'8-Spd TorqueFlite Automatic',
                                       'Automatic Transmission':'8-Speed Automatic',
                                       '8-SPEED AUTOMATIC (845RE)':'8-Speed Automatic (845RE)',
                                       '8-speed Automatic':'8-Speed Automatic',
                                       '8 Spd Automatic':'8-Speed Automatic',
                                       'Automanual':'8-Speed Shiftable Automatic'}, inplace=True)
#for column in test_data.columns:
    #if test_data[column].dtype == 'object':
        #print(test_data[column].value_counts())
        #print("\n")
test_data = unknown_and_encoder(test_data)
#print(test_data.isnull().sum())
X_test = pd.get_dummies(test_data, columns=columns_to_dummy)
# Confirming Training and Testing are the same shape
#print(len(X.columns))
#print(X_test.columns)
y_pred_reg = rf_reg.predict(X_test)
#print("First 10 predictions:", y_pred_reg[:10])
y_pred_class = rf_class.predict(X_test)
#print("First 10 predictions:", y_pred_class[:10])
output = test_data[['ListingID']]
output['PredictedTrim'] = y_pred_class
output['PredictedPrice'] = y_pred_reg
#print(output)
#output.to_csv('Output_data.csv', index=False)
#output_test = pd.read_csv('Output_data.csv')
#print(output_test)
end_time = time.time()
print(end_time - start_time, 'seconds')