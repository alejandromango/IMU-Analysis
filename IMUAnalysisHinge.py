import numpy
numpy.seterr(divide='warn')
import pandas
from tkinter import Tk
from tkinter.filedialog import askdirectory

def getJEst(aEst):
    j1Est = numpy.array([numpy.float(numpy.cos(aEst['angle1'])*numpy.cos(aEst['azimuth1'])), numpy.float(numpy.cos(aEst['angle1'])*numpy.sin(aEst['azimuth1'])), numpy.float(numpy.sin(aEst['angle1']))])
    j2Est = numpy.array([numpy.float(numpy.cos(aEst['angle2'])*numpy.cos(aEst['azimuth2'])), numpy.float(numpy.cos(aEst['angle2'])*numpy.sin(aEst['azimuth2'])), numpy.float(numpy.sin(aEst['angle2']))])
    return numpy.array([j1Est, j2Est])

def getGRates(g, delta):
    r = pandas.DataFrame({})
    # for i in g.columns.values.tolist():
    #     # Create time-shifted arrays for calculating gyroscope rate with third order approx
    #     gn2 = g[i].values[:-4]
    #     gn1 = g[i].values[1:-3]
    #     gp1 = g[i].values[3:-1]
    #     gp2 = g[i].values[4:]
    #     # Third order approximation of derivative wrt time
    #     r[i] = (gn2 - 8*gn1 + 8*gp1 - gp2)/(12*delta)
    return r

def getGErrorVector(j, g):
    left = numpy.linalg.norm(numpy.cross(j[0], g[0]), axis=1)
    right = numpy.linalg.norm(numpy.cross(j[1], g[1]), axis=1)
    errs = left - right
    return(errs)

def getSumSquares(j, g):
    left = numpy.linalg.norm(numpy.cross(g[0], j[0]), axis=1)
    right = numpy.linalg.norm(numpy.cross(g[1], j[1]), axis=1)
    errs = left - right
    return(numpy.sum(errs**2))

def getGJacobian(j, g):

    numerator1 = numpy.cross(numpy.cross(g[0], j[0]), g[0])
    denominator1 = numpy.linalg.norm(numpy.cross(g[0], j[0]), axis=1)
    grad1 = numpy.divide(numerator1.T, denominator1).T
    numerator2 = numpy.cross(numpy.cross(g[1], j[1]), g[1])
    denominator2 = numpy.linalg.norm(numpy.cross(g[1], j[1]), axis=1)
    grad2 = numpy.divide(numerator2.T, denominator2).T
    
    return(numpy.array([grad1, grad2]))

# Get directories for IMU data
forearmPath = "/Users/alex/Library/Mobile Documents/com~apple~CloudDocs/School/Masters/BME_207/Term Project/IMU Analysis/ForearmMovement"#askdirectory(title='Select Folder Containing Forearm Data') # shows dialog box and return the path
print(forearmPath)
armPath = "/Users/alex/Library/Mobile Documents/com~apple~CloudDocs/School/Masters/BME_207/Term Project/IMU Analysis/UpperArmMovementTimeMod"#askdirectory(title='Select Folder Containing Arm Data') # shows dialog box and return the path
print(armPath)
foreArmGData = pandas.read_csv("{}/Gyroscope.csv".format(forearmPath))
armGData = pandas.read_csv("{}/Gyroscope.csv".format(armPath))
# Truncate data to have same number of samples (temporary with contrived data)
minDataLength = min(len(armGData), len(foreArmGData))
foreArmGData = foreArmGData[:minDataLength]
armGData = armGData[:minDataLength]
# Calculate time delta between samples in seconds
deltat = numpy.average(numpy.diff(foreArmGData['Time (s)'].values))
# Vector of initial guess for angle and azimuth of joint axis in coordinates of both sensors
aaVec = pandas.DataFrame({'angle1': [1.1], 'azimuth1': [1.0], 'angle2': [1.2], 'azimuth2': [1.4]})
# Initialize gyroscope data array(g1, g2) (2 arrays with # rows equal to # sample and each row [x, y, z])
gData = numpy.array([numpy.stack([foreArmGData['X (rad/s)'].values, foreArmGData['Y (rad/s)'].values, foreArmGData['Z (rad/s)'].values], 1),\
                    numpy.stack([armGData['X (rad/s)'].values, armGData['Y (rad/s)'].values, armGData['Z (rad/s)'].values], 1)])
# Initialize dataframe for initial joint
jEst = getJEst(aaVec)
# Intialize gyroscope rates data vectors in dataframe
oldJEst = []
for i in range(10):
    # Get error vector
    gErrors = getGErrorVector(jEst, gData)
    print(getSumSquares(jEst, gData))
    #Calculate Jacobian
    gJacobian = getGJacobian(jEst, gData)
    # Calculate pseudoinverce
    gPseudoinverse = numpy.linalg.pinv(gJacobian)
    a = - numpy.dot(gPseudoinverse, gErrors)
    oldJEst.append(jEst)
    jEst = jEst - numpy.dot(gPseudoinverse, gErrors)
