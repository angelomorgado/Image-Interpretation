def numberOfPeople(number_of_people, image_name):
    if number_of_people == 0:
        return f"There isn't a single person in the image {image_name}, so it is impossible to extract any features. "
    elif number_of_people == 1:
        return f"There is only one person in the image {image_name}. "
    elif number_of_people == 2:
        return f"There are two people in the image {image_name}. "
    else:
        return f"There are {number_of_people} people in the image {image_name}. "
    
# Function that returns the text based on the number of people facing the camera
# people_orientation : [cameraCount, backCount, sideCount]
def cameraOrientation(number_of_people, people_orientation):
    if number_of_people == 0:
        return ''
    elif number_of_people == 1:
        if people_orientation[0] == 1:
            return 'The person is facing the camera. '
        elif people_orientation[1] == 1:
            return 'The person is facing the opposite direction to the camera. '
        else:
            return 'The person is sideways to the camera. '
    else:
        s = ''
        
        # Facing the camera
        if people_orientation[0] == 0:
            s += 'None are facing the camera. '
        elif people_orientation[0]/number_of_people < 1:
            s += f'Some of them are facing the camera, namely {people_orientation[0]}, '
        else:
            s += f'All of them are facing the camera, '

        # Facing the opposite direction to the camera
        if people_orientation[1] == 0:
            s += 'while no one have their backs to the camera, '
        elif people_orientation[1]/number_of_people < 1:
            s += f'while {people_orientation[1]} have their backs to the camera, '    
        else:
            s += f'while all of them have their backs to the camera, '
            
        # Sideways to the camera
        if people_orientation[2] == 0:
            s += 'and none are sideways. '
        elif people_orientation[2]/number_of_people < 1:
            s += f'and {people_orientation[2]} are sideways. '
        else:
            s += f'and all of them are sideways. '    
            
        return s


# Function that returns the text based on the number of people standing/sitting/laying
# people_pose : [standingCount, sittingCount, layingCount]
def peoplePose(number_of_people, people_pose):
    if number_of_people == 0:
        return ''
    elif number_of_people == 1:
        if people_pose[0] == 1:
            return 'The person is standing. '
        elif people_pose[1] == 1:
            return 'The person is sitting. '
        else:
            return 'The person is laying. '
    else:
        s = ''
        
        # People standing
        if people_pose[0] == 0:
            s += 'Not a single person is standing. '
        elif people_pose[0]/number_of_people < 1:
            s += f'Some of them are standing, about {people_pose[0]}, '
        else:
            s += 'Every person is standing, '
        
        # People sitting
        if people_pose[1] == 0:
            s += 'whereas no one is sitting, '
        elif people_pose[1]/number_of_people < 1:
            s += f'whereas {people_pose[1]} are sitting, '
        else:
            s += 'whereas all of them are sitting, '
            
        # People laying
        if people_pose[2] == 0:
            s += 'and none are laying. '
        elif people_pose[2]/number_of_people < 1:
            s += f'and {people_pose[1]} are laying. '
        else:
            s += 'and every single one of them is laying. '
        
        return s
        

# Function that returns the text based on the number of people close/far from the camera
# people_distance : [closeCount, farCount]
def peopleDistance(number_of_people, people_distance):
    if number_of_people == 0:
        return ''
    elif number_of_people == 1:
        if people_distance[0] == 1:
            return 'The person is close to the camera. '
        else:
            return 'The person is far from the camera. '
    else:
        s = ''
        
        # Close to the camera
        if people_distance[0] == 0:
            s += 'None are close to the camera. '
        elif people_distance[0]/number_of_people < 1:
            s += f'A few of them are close to the camera, about {people_distance[0]}, '
        else:
            s += 'All of them are close to the camera, '
        
        # Far from the camera
        if people_distance[1] == 0:
            s += 'meanwhile no one is far from the camera. '
        elif people_distance[1]/number_of_people < 1:
            s += f'meanwhile {people_distance[1]} are far from the camera. '
        else:
            s += 'meanwhile all of them are far from the camera. '
        
        return s

# Function that returns the text based on the number of people with/without their face covered (possibly by a mask)
# number_masks: number of people with masks
def peopleMasked(number_of_people, number_masks):
    if number_of_people == 0:
        return ''
    elif number_of_people == 1:
        if number_masks > 0:
            return 'The person has their face covered (possibly by a mask). '
        else:
            return 'The person does not have their face covered (possibly by a mask). '
    else:
        if number_masks == 0:
            return 'No one has their face covered (possibly by a mask). '
        elif number_masks/number_of_people < 1:
            return f'Some people have their faces covered (possibly by a mask), more precisely {number_masks} of them. '
        else :
            return 'Every person that has their face covered (possibly by a mask). '
        
def shirtTone(number_of_people, shirt_tone):
    if number_of_people == 0:
        return ''
    elif number_of_people == 1:
        if shirt_tone == 'dark':
            return 'The individual is wearing a dark shirt, '
        else:
            return 'The shirt the person is wearing is light colored, '
    else:
        if shirt_tone == 'dark':
            return 'The most prominent shirt tone is dark, '
        elif shirt_tone == 'light':
            return 'Most individuals in the image are wearing light colored shirts, '
        else:
            return 'The distribution of the tone in shirts is pretty even, '
        
def pantsTone(number_of_people, pants_tone):
    if number_of_people == 0:
        return ''
    elif number_of_people == 1:
        if pants_tone == 'dark':
            return 'and their pants are dark. '
        else:
            return 'and the tone of their pants is light. '
    else:
        if pants_tone == 'dark':
            return 'and dark colored pants are worn by most of the people. '
        elif pants_tone == 'light':
            return 'and light colored pants are more used by most of the people. '
        else:
            return 'and the pants tonality is equilibrated. '
