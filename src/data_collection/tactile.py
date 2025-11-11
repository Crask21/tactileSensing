import rospy
from sr_robot_msgs.msg import MSTAll

class TactileReader:
    def __init__(self):
        # Initialize the ROS node
        if not rospy.core.is_initialized():
            rospy.init_node('tactile_reader', anonymous=True)
        
        # Placeholder for the latest tactile data
        self.latest_data = None
        
        # Subscribe to the /rh/tactile topic
        self.subscriber = rospy.Subscriber('/rh/tactile', MSTAll, self.callback)

        # Create a header for the data structure
        self.header = [f"{finger}_{sensor}_{axis}" for finger in ['th', 'ff', 'mf', 'rf', 'lf']
                   for sensor in range(1, 18) for axis in ['x', 'y', 'z']]
        

    def callback(self, data):
        # Callback function to store the latest tactile data
        self.latest_data = data
    
    def read(self):
        # Return the latest tactile data
        if self.latest_data is None:
            rospy.logwarn("No tactile data received yet.")


        list = []
        for i in range(self.latest_data.tactiles.__len__()):
            for j in range(self.latest_data.tactiles[i].magnetic_data.__len__()):
                list.append((self.latest_data.tactiles[i].magnetic_data[j].x, self.latest_data.tactiles[i].magnetic_data[j].y, self.latest_data.tactiles[i].magnetic_data[j].z))
        
        # Flatten the list of tuples
        list = [coordinate for sublist in list for coordinate in sublist]
        return list

if __name__ == "__main__":
    # Create an instance of the TactileReader
    reader = TactileReader()
    
    # Wait for tactile data and print it
    rospy.loginfo("Waiting for tactile data...")
    rospy.sleep(1)  # Allow some time for data to be received
    tactile_data = reader.read()
    
    if tactile_data:
        rospy.loginfo("Tactile data received:")
        print(tactile_data)
        print(len(tactile_data))
    else:
        rospy.logwarn("No tactile data available.")