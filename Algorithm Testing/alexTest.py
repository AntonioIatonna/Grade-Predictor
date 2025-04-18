import pandas as pd

data = pd.read_csv('CombiDataset.csv')

train = data.sample(frac=0.75, random_state=42) 
  
# Dropping all those indexes from the dataframe that exists in the train_set 
test = data.drop(train.index) 
train.shape, test.shape

labsTotal=0
assignTotal=0
midTotal=0
labsVariance=0
assignVariance=0
midVariance=0

for index, row in train.iterrows():
    labTemp=(row['L1']+row['L2']+row['L3']+row['L4']+row['L5']+row['L6']+row['L7']+row['L7B']+row['L8'])/9 - row['F']
    assignTemp=(row['A1']+row['A2']+row['A3']+row['A4']+row['A5'])/5 - row['F']
    midTemp=row['M'] - row['F']
    
    labsTotal+=labTemp
    assignTotal+=assignTemp
    midTotal+=midTemp
    labsVariance+=(labTemp**2)
    assignVariance+=(assignTemp**2)
    midVariance+=(midTemp**2)

labsAVG=labsTotal/train.shape[0]
assignAVG=assignTotal/train.shape[0]
midAVG=midTotal/train.shape[0]
print("Averages Differences: ",labsAVG,assignAVG,midAVG)

labsVariance=labsVariance/(train.shape[0]-1)
assignVariance=assignVariance/(train.shape[0]-1)
midVariance=midVariance/(train.shape[0]-1)
print("Variances: ",labsVariance,assignVariance,midVariance)

labsPortion=(1/labsVariance)/(1/labsVariance+1/assignVariance+1/midVariance)
assignPortion=(1/assignVariance)/(1/labsVariance+1/assignVariance+1/midVariance)
midtermPortion=(1/midVariance)/(1/labsVariance+1/assignVariance+1/midVariance)
print("portions: ",labsPortion,assignPortion,midtermPortion)

# Test portion 

finalGuess = 0
guessDiff=0
guessDiffSquared=0
diff_less_than_1 = 0
diff_less_than_5 = 0
diff_less_than_10 = 0
diff_less_than_20 = 0

for index, row in test.iterrows():
    labTemp=(row['L1']+row['L2']+row['L3']+row['L4']+row['L5']+row['L6']+row['L7']+row['L7B']+row['L8'])/9 - labsAVG
    assignTemp=(row['A1']+row['A2']+row['A3']+row['A4']+row['A5'])/5 - assignAVG
    midTemp=row['M'] - midAVG

    finalGuess = labsPortion*labTemp + assignTemp*assignPortion + midTemp * midtermPortion
    
    thisGuessDifference = finalGuess-row['F']
    guessDiff+=abs(thisGuessDifference)
    guessDiffSquared+=(abs(thisGuessDifference)**2)

    thisGuessDifferenceABS = abs(thisGuessDifference)

    if thisGuessDifferenceABS <= 1:
        diff_less_than_1 += 1
    if thisGuessDifferenceABS <= 5:
        diff_less_than_5 += 1
    if thisGuessDifferenceABS <= 10:
        diff_less_than_10 += 1
    if thisGuessDifferenceABS <= 20:
        diff_less_than_20 += 1
    
guessDiff = guessDiff/test.shape[0]
guessDiffSquared = guessDiffSquared/test.shape[0]
print("Out of", test.shape[0],"tests")  
print("Predictions within 1%:", diff_less_than_1)
print("Predictions within 5%:", diff_less_than_5)
print("Predictions within 10%:", diff_less_than_10)
print("Predictions within 20%:", diff_less_than_20)
print("Average difference: ",guessDiff)
print("Average difference: ",guessDiffSquared)