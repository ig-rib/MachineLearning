#!/bin/python3

# P(joven | 1, no 2, 3, no 4) = P(1, no 2, 3, no 4 | joven) * P(joven) / P(1, 3) = 
# P(1 | joven) * P(no 2 | joven) * P(3 | joven) * P(no 4 | joven) / ( P(1, no 2, 3, no 4 | joven) * P(joven)  + P(1, no 2, 3, no 4 | viejo) * P(viejo) )=
pJoven = ((0.95*0.95*0.2*0.8)*0.1)/((.95*.95*0.2*0.8)*0.1 + (0.03*0.18*0.34*0.08)*0.9)
print('Probabilidad de que sea joven: ', pJoven)
# P(viejo | 1, no 2, 3, no 4) = P(1, no 2, 3, no 4 | viejo) * P(viejo) / P(1, 3) = 
# P(1 | viejo) * P(no 2 | viejo) * P(3 | viejo) * P(no 4 | viejo) / ( P(1, no 2, 3, no 4 | joven) * P(joven)  + P(1, no 2, 3, no 4 | viejo) * P(viejo) )=
pViejo = ((0.03*0.18*0.34*0.08)*0.9)/((.95*.95*0.2*0.8)*0.1 + (0.03*0.18*0.34*0.08)*0.9)
print('Probabilidad de que sea viejo: ', pViejo)