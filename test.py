from parser import Parser

def test(filename):
    p = Parser(filename)
    X = p.transform_x()
    Y = p.transform_y()

    t_success = 0
    val_success = [0,0]
    count = [0,0]
    for i in range(len(X)):
        yi = Y[i]
        xi = X[i]
        res = result(hypothesis(xi, W))
        if res == yi:
            t_success += 1 
            val_success[yi] += 1
        count[yi] += 1
            
    print ('Correct: {}'.format(t_success * 100 / len(X)))
    print ('Positive correct {}'.format(val_success[1] * 100 / count[1]))
    print ('Negative correct {}'.format(val_success[0] * 100 / count[0]))
            
