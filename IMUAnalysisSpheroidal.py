import numpy
import pandas
from tkinter import Tk
from tkinter.filedialog import askdirectory

def gamma(g, a, o):
    a = 1

def gammaTranspose(g, a, o):
    numpy.cross(numpy.cross(g, o))

def getJEst(aEst):
    j1Est = numpy.array([numpy.float(numpy.cos(aEst['angle1'])*numpy.cos(aEst['azimuth1'])), numpy.float(numpy.cos(aEst['angle1'])*numpy.sin(aEst['azimuth1'])), numpy.float(numpy.sin(aEst['angle1']))])
    j2Est = numpy.array([numpy.float(numpy.cos(aEst['angle2'])*numpy.cos(aEst['azimuth2'])), numpy.float(numpy.cos(aEst['angle2'])*numpy.sin(aEst['azimuth2'])), numpy.float(numpy.sin(aEst['angle2']))])
    return pandas.DataFrame({'j1': j1Est, 'j2': j2Est})

def getGRates(g, delta):
    r = pandas.DataFrame({})
    for i in g.columns.values.tolist():
        # Create time-shifted arrays for calculating gyroscope rate with third order approx
        gn2 = g[i].values[:-4]
        gn1 = g[i].values[1:-3]
        gp1 = g[i].values[3:-1]
        gp2 = g[i].values[4:]
        # Third order approximation of derivative wrt time
        r[i] = (gn2 - 8*gn1 + 8*gp1 - gp2)/(12*delta)
    return r

def getGErrorVector(j, g):
    g1 = numpy.array([g['g1x'], g['g1y'], g['g1z']])
    g2 = numpy.array([g['g2x'], g['g2y'], g['g2z']])
    errs = []
    for i in range(len(g['g1x'])):
        left = numpy.linalg.norm(numpy.cross(g1[:, i], j['j1']))
        right = numpy.linalg.norm(numpy.cross(g2[:, i], j['j2']))
        errs.append(left - right)
    return(numpy.array(errs))

def getGJacobian(j, g):
    g1 = numpy.array([g['g1x'], g['g1y'], g['g1z']])
    g2 = numpy.array([g['g2x'], g['g2y'], g['g2z']])
    grad1 = []
    grad2 = [] 
    for i in range(len(g['g1x'])):
        numerator1 = numpy.cross(numpy.cross(g1[:, i], j['j1']), g1[:, i])
        denominator1 = numpy.linalg.norm(numpy.cross(g1[:, i], j['j1']))
        grad1.append(numerator1/denominator1)
        numerator2 = numpy.cross(numpy.cross(g2[:, i], j['j2']), g2[:, i])
        denominator2 = numpy.linalg.norm(numpy.cross(g2[:, i], j['j2']))
        grad2.append(numerator2/denominator2)
    
    return({'jac1': numpy.array(grad1), 'jac2': numpy.array(grad2)})

# Get directories for IMU data
forearmPath = "/Users/alex/Library/Mobile Documents/com~apple~CloudDocs/School/Masters/BME_207/Term Project/IMU Analysis/ForearmMovement"#askdirectory(title='Select Folder Containing Forearm Data') # shows dialog box and return the path
print(forearmPath)
armPath = "/Users/alex/Library/Mobile Documents/com~apple~CloudDocs/School/Masters/BME_207/Term Project/IMU Analysis/UpperArmMovementTimeMod"#askdirectory(title='Select Folder Containing Arm Data') # shows dialog box and return the path
print(armPath)
foreArmGData = pandas.read_csv("{}/Gyroscope.csv".format(forearmPath))
armGData = pandas.read_csv("{}/Gyroscope.csv".format(armPath))
foreArmAData = pandas.read_csv("{}/Linear Accelerometer.csv".format(forearmPath))
armAData = pandas.read_csv("{}/Linear Accelerometer.csv".format(armPath))
# Truncate data to have same number of samples (temporary with contrived data)
minDataLength = 10#min(len(armGData), len(foreArmGData))
foreArmGData = foreArmGData[:minDataLength]
armGData = armGData[:minDataLength]
foreArmAData = foreArmAData[:minDataLength]
armAData = armAData[:minDataLength]
# Calculate time delta between samples in seconds
deltat = numpy.average(numpy.diff(foreArmGData['Time (s)'].values))
# Vector of initial guesses for offset vectors (o1, o2) for each sensor (pointing to joint center from sensor center)
oVec = numpy.array([[1, 2,1], [2, 1, 2]])
# Initialize gyroscope data array(g1, g2) (2 arrays with # rows equal to # sample and each row [x, y, z])
gData = numpy.array([numpy.stack([foreArmGData['X (rad/s)'].values, foreArmGData['Y (rad/s)'].values, foreArmGData['Z (rad/s)'].values], 1),\
                    numpy.stack([armGData['X (rad/s)'].values, armGData['Y (rad/s)'].values, armGData['Z (rad/s)'].values], 1)])
# Initialize accelerometer data array (a1, a1) (2 arrays with # rows equal to # samples and each row [x, y, z])
aData = numpy.array([numpy.stack([foreArmAData['X (m/s^2)'].values, foreArmAData['Y (m/s^2)'].values, foreArmAData['Z (m/s^2)'].values], 1),\
                    numpy.stack([armAData['X (m/s^2)'].values, armAData['Y (m/s^2)'].values, armAData['Z (m/s^2)'].values], 1)])
# Calculate third-order approximation for gyroscope derivatives
# Initialize dataframe for initial joint
jEst = getJEst(aaVec)
# Intialize gyroscope rates data vectors in dataframe
gRates = getGRates(gData, deltat)
for i in range(10):
    # Get error vector
    gErrors = getGErrorVector(jEst, gData)
    print(numpy.sum(gErrors**2))
    #get jacobian
    gJacobian = getGJacobian(jEst, gData)
    gPseudoinverse = {'ps1': numpy.linalg.pinv(gJacobian['jac1']), 'ps2': numpy.linalg.pinv(gJacobian['jac2'])}
    # Something weird here
    jEst['j1'] = jEst['j1'] + numpy.dot(gPseudoinverse['ps1'], -gErrors)
    jEst['j2'] = jEst['j2'] + numpy.dot(gPseudoinverse['ps2'], -gErrors)
    # gUpdater = numpy.array([gPseudoinverse['ps1'] @ gErrors, gPseudoinverse['ps2'] @ gErrors])
    # jEst = jEst - numpy.transpose(gUpdater)
