import numpy as np
from scipy.spatial import distance
from PIL import Image
import imageio

#============================= Auxiliary functions ===========================================
# Returns a tuple containing the chosen body part's informations
def __getBodyPart(n, person_keypoints):
    x = person_keypoints[n * 3 - 3] # x axis
    y = person_keypoints[n * 3 - 2] # y axis
    v = person_keypoints[n * 3 - 1] # visibility
    return (x,y,v)

# returns the angle between three points assuming point B is the center (knee)
def __getAngleThreePoints(pointA, pointB, pointC):
    ba = (pointA[0] - pointB[0], pointA[1] - pointB[1])
    bc = (pointC[0] - pointB[0], pointC[1] - pointB[1])
    cosine_angle = np.dot(ba,bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)    

# Returns the point that is between two points
def __getMidpoint(pointA, pointB):
    return ((pointA[0] + pointB[0])/2, (pointA[1] + pointB[1])/2)

#============================= Pose functions ===========================================

# When sitting facing the camera
# If the ratio is bigger than alpha it means the person is sitting
def __getLegRatio(hip, knee, ankle):
    dist_hip_knee = distance.euclidean(hip[:2], knee[:2]) #__euclidianDistance(hip[:2], knee[:2])
    dist_knee_ankle = distance.euclidean(knee[:2], ankle[:2])
    alpha = 1.5
    return dist_knee_ankle / dist_hip_knee >= alpha

# When sitting with an angle to the camera
# If the angle is between minimum_alpha and maximum_alpha it means the person is sitting    
def __getLegAngle(hip, knee, ankle):
    minimum_alpha = 40
    maximum_alpha = 120
    angle = __getAngleThreePoints(hip[:2], knee[:2], ankle[:2])   
    return angle <= maximum_alpha and angle >= minimum_alpha

def __getBodyAngle(shoulder, hip, knee):
    minimum_alpha = 40
    maximum_alpha = 120
    angle = __getAngleThreePoints(shoulder[:2], hip[:2], knee[:2])
    return angle <= maximum_alpha and angle >= minimum_alpha
   
def __getLayingPose(person_keypoints):
    minimum_alpha = 0.9
    maximum_alpha = 1.1
    
    left_shoulder = __getBodyPart(6, person_keypoints)
    right_shoulder = __getBodyPart(7, person_keypoints)
    left_hip = __getBodyPart(12, person_keypoints)
    right_hip = __getBodyPart(13, person_keypoints)
    
    middle_hips = __getMidpoint(left_hip[:2], right_hip[:2])
    middle_shoulders = __getMidpoint(left_shoulder[:2], right_shoulder[:2])
    
    return middle_shoulders[1] / middle_hips[1] < maximum_alpha and middle_shoulders[1] / middle_hips[1] > minimum_alpha
    
def __getSittingPose(person_keypoints):
    # Right side
    right_shoulder = __getBodyPart(7, person_keypoints)
    right_hip = __getBodyPart(13, person_keypoints)
    right_knee = __getBodyPart(15, person_keypoints)
    right_ankle = __getBodyPart(17, person_keypoints)
    
    # Left side
    left_shoulder = __getBodyPart(6, person_keypoints)
    left_hip = __getBodyPart(12, person_keypoints)
    left_knee = __getBodyPart(14, person_keypoints)
    left_ankle = __getBodyPart(16, person_keypoints)
    
    # If the visibility of the body parts is less than 0.5 it means that the body part is not in the frame
    if (left_knee[2] < 0.5 and right_knee[2] < 0.5) or  (left_ankle[2] < 0.5 and right_ankle[2] < 0.5) or  (left_hip[2] < 0.5 and right_hip[2] < 0.5):
        return False
    
    return __getLegRatio(left_hip, left_knee, left_ankle) or __getLegRatio(right_hip, right_knee, right_ankle) \
    or __getLegAngle(left_hip, left_knee, left_ankle) or __getLegRatio(left_hip, left_knee, left_ankle)  \
    or __getBodyAngle(left_shoulder, left_hip, left_knee) or __getBodyAngle(right_shoulder, right_hip, right_knee)
       
#============================= Interpretability functions ===================================

def getPeopleQuantity(pred):
    return len(pred)

def getPeopleOrientation_maskedFaces(imgName, alphapose):
    cameraCount = 0
    backCount = 0
    sideCount = 0
    
    maskedCount = 0
    maskedAlpha = 0.3
    
    for person in alphapose:
        if person['image_id'] == imgName:
            left_shoulder = __getBodyPart(6, person['keypoints'])
            right_shoulder = __getBodyPart(7, person['keypoints'])
            left_hip = __getBodyPart(12, person['keypoints'])
            right_hip = __getBodyPart(13, person['keypoints'])
            
            # facing the camera
            if left_shoulder[0] > right_shoulder[0] and left_hip[0] > right_hip[0]:
                cameraCount += 1
                
                # Get covered face information
                nose = __getBodyPart(1, person['keypoints'])
                if nose[2] < maskedAlpha:
                    maskedCount += 1
            # back to the camera
            elif left_shoulder[0] < right_shoulder[0] and left_hip[0] < right_hip[0]:
                backCount += 1
            # side to the camera
            else:
                sideCount += 1
    
    return cameraCount, backCount, sideCount, maskedCount
    
def getPeoplePose(imgName, alphapose):
    standingCount = 0
    sittingCount = 0
    layingCount = 0
    
    for person in alphapose:
        if person['image_id'] == imgName:
            if __getLayingPose(person['keypoints']):
                layingCount += 1
            elif __getSittingPose(person['keypoints']):
                sittingCount += 1
            else:
                standingCount += 1
            
    return standingCount, sittingCount, layingCount
    
def getPeopleDistance(segmentation):
    farCount = 0
    closeCount = 0
    area = []
    
    for person in segmentation:
        area.append((person == 1).sum() / person.size)
    
    area = np.array(area)  
    
    with open('AuxFiles/areaValue.txt', 'r') as f:
        alpha = float(f.read().split('\n')[0])
    
    closeCount += (area > alpha).sum()
    farCount += (area < alpha).sum()
       
    return farCount, closeCount

def getShirtTones(imgName, alphapose):
    darkCount = 0
    lightCount = 0
    imgPath = '../EvaluationDataset/seg_input/'
    
    for person in alphapose:
        if person['image_id'] == imgName:
            left_shoulder = __getBodyPart(6, person['keypoints'])
            right_shoulder = __getBodyPart(7, person['keypoints'])
            left_hip = __getBodyPart(12, person['keypoints'])
            right_hip = __getBodyPart(13, person['keypoints'])
            
            # Shoulders mid point
            shoulders_midpoint = __getMidpoint(left_shoulder[:2], right_shoulder[:2])
            
            # Hips mid point
            hips_midpoint = __getMidpoint(left_hip[:2], right_hip[:2])
            
            # Torso mid point
            torso_midpoint = __getMidpoint(shoulders_midpoint, hips_midpoint)
            x = int(torso_midpoint[0])
            y = int(torso_midpoint[1])
            
            img = np.array(Image.open(imgPath + imgName))
            
            if img[y][x].sum() < 128*3:
                darkCount += 1
            else:
                lightCount += 1
                
    return darkCount, lightCount

def getJeanTones(imgName, alphapose):
    darkCount = 0
    lightCount = 0
    imgPath = '../EvaluationDataset/seg_input/'
    
    for person in alphapose:
        if person['image_id'] == imgName:
            left_hip = __getBodyPart(12, person['keypoints'])
            left_knee = __getBodyPart(14, person['keypoints'])
            
            # Shoulders mid point
            leg_midpoint = __getMidpoint(left_hip[:2], left_knee[:2])
            
            x = int(leg_midpoint[0])
            y = int(leg_midpoint[1])
            
            img = np.array(Image.open(imgPath + imgName))
            
            if img[y][x].sum() < 128*3:
                darkCount += 1
            else:
                lightCount += 1
                
    return darkCount, lightCount
            