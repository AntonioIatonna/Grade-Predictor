def outPutTestignData(y_test,y_pred):
    i=0
    total_difference=0
    mse=0
    diff_less_than_20=0
    diff_less_than_10=0
    diff_less_than_5=0
    diff_less_than_1=0


    for x in y_pred:
        diff = abs(y_pred[i] - y_test.iloc[i])
        total_difference = total_difference + diff
        diff_squared = diff * diff
        mse = mse + diff_squared

        if diff<=1:
            diff_less_than_1 = diff_less_than_1 + 1

        if diff<=5:
            diff_less_than_5 = diff_less_than_5 + 1

        if diff<=10:
            diff_less_than_10 = diff_less_than_10 + 1
        
        if diff<=20:
            diff_less_than_20 = diff_less_than_20 + 1

        i = i + 1

    print("RESULTS: out of", len(y_test), "testing data")
    print("Predictions within 1%:", diff_less_than_1)
    print("Predictions within 5%:", diff_less_than_5)
    print("Predictions within 10%:", diff_less_than_10)
    print("Predictions within 20%:", diff_less_than_20)
    print("Average difference between predicted and real:", total_difference / len(y_test))
    print("MSE:", mse / len(y_test))

def checkWithin5(y_test,y_pred) -> int:
    i=0
    diff_less_than_5=0

    for x in y_pred:
        diff = abs(y_pred[i] - y_test.iloc[i])

        if diff<=5:
            diff_less_than_5 = diff_less_than_5 + 1

        i = i + 1

    return diff_less_than_5


