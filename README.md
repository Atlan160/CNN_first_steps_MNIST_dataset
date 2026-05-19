erste Schritte.py
Hier kann man die Grundlagen des Trainings ausprobieren und testen wie sich das Neuronale Netz (NN) verhält wenn man z.B. die Parameter drop_out, epochs, learning rate oder hidden layers dimension.
Am Ende wird immer gegen den Testsatz an Bildern getestet um zu gucken ob das Modell gut auf unbekannten Daten performt (wichtig um Overfitting zu vermeiden).

Optimierung mit Optuna.py
Hier wurde der Code aus erste Schritte.py genommen und so verändert das man den Code iterativ durchlaufen und das Ergebnis auswerten kann. So kann man mit Optuna Parameter finden welche optimal für das NN passen.
!Sehr rechenaufwendig, mehr wie 10 Trials nur mit aktueller Grafikkarte zu empfehlen!

optuna output.txt
Der Output von Optimierung mit Optuna.py für 20 Trials. Am Ende wurde ein guter Parametersatz gefunden welcher für test gegen Eigene Bilder.py genutzt werden kann.

test gegen Eigene Bilder.py
Das Modell (.pth Datei) muss einmal trainiert werden und kann dann, wenn es nicht verändert wird, immer wieder geladen werden. Ist der Trainings oder Ladeprozess abgeschlossen kann man im sich
öffnenden Zeichenfenster eigene Zahlen zeichnen und danach vom Modell evaluieren lassen. Es werden die Softmax Wahrscheinlichkeiten für jede Zahl angezeigt und das 28x28 Pixel Bild welches dem MOdell übergeben wird.

Spannend ist hier Zahlen nur halb fertig zu zeichnen, diese Stückweise weiterzuzeichnen und zu schauen ab wann das NN die Zahl als korrekt erkennt.
