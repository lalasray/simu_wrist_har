import os

# Define the directory containing the BVH files
directory = "/home/lala/other/Repos/git/simu_wrist_har/data/how2sign/val/bvh/"

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".bvh"):  # Ensure it's a BVH file
        file = os.path.join(directory, filename)
        
        # Your existing script here
        import bpy 
        from mathutils import Vector, Matrix
        import csv
        
        bpy.ops.import_anim.bvh(filepath=file)
        bpy.context.object.rotation_euler[1] = 3.14159
        bpy.context.object.scale[0]  = 0.01
        bpy.context.object.scale[1]  = 0.01
        bpy.context.object.scale[2]  = 0.01
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        
        obj = bpy.context.active_object
        action = obj.animation_data.action
        frame_range = action.frame_range
        start_frame, end_frame = frame_range
        
        armature_name = filename.split('.')[0]
        
        # Gravity vector
        gravity = Vector((0, 0, -9.8)) 
        
        def analyze_bone_motion(bone_name, start_frame, end_frame):
            # Get the armature and bone objects
            armature_obj = bpy.data.objects[armature_name]
            bone_obj = armature_obj.pose.bones[bone_name]
        
            # Set up variables to store previous data
            previous_location = None
            previous_velocity = None
            previous_orientation = None
        
            # Open a new CSV file for writing
            with open('/home/lala/other/Repos/git/simu_wrist_har/data/how2sign/val/imu/'+ str(armature_name) + '_' +str(bone_name) + '.csv', mode='w', newline='') as imu_file:
                # Create a CSV writer object
                imu_writer = csv.writer(imu_file)
        
                # Write the header row
                imu_writer.writerow(['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        
                # Loop over the keyframes
                for frame in range(int(start_frame), int(end_frame) + 1):
                    # Set the current frame
                    bpy.context.scene.frame_set(frame)
        
                    # Get the local bone matrix
                    bone_matrix = bone_obj.matrix
        
                    # Get the global bone matrix
                    bone_matrix = bone_obj.matrix @ armature_obj.matrix_world
        
                    # Get the global location and orientation of the bone
                    location = bone_matrix.to_translation()
                    orientation = bone_matrix.to_quaternion()
        
                    # Calculate the time step
                    time_step = 1.0 / bpy.context.scene.render.fps
        
                    # Calculate the velocity
                    if previous_location is not None:
                        displacement = location - previous_location
                        velocity = displacement / time_step
                        acceleration = (velocity - previous_velocity) / time_step
                    else:
                        velocity = Vector((0, 0, 0))
                        acceleration = Vector((0, 0, 0))
                    previous_location = location
                    previous_velocity = velocity
        
                    # Calculate the angular velocity and acceleration
                    if previous_orientation is not None:
                        angular_displacement = orientation.rotation_difference(previous_orientation)
                        angular_velocity = angular_displacement.axis * angular_displacement.angle / time_step
                        angular_acceleration = (angular_velocity - previous_angular_velocity) / time_step
                    else:
                        angular_velocity = Vector((0, 0, 0))
                        angular_acceleration = Vector((0, 0, 0))
                    previous_orientation = orientation
                    previous_angular_velocity = angular_velocity
        
                    # Apply gravity to the acceleration
                    local_gravity = bone_obj.matrix @ gravity
                    acceleration -= local_gravity
        
                    # Generate and write the data rows
                    timestamp = frame - 1
                    accel_x = acceleration.x
                    accel_y = acceleration.y
                    accel_z = acceleration.z
                    gyro_x = angular_velocity.x
                    gyro_y = angular_velocity.y
                    gyro_z = angular_velocity.z
        
                    imu_writer.writerow([timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
        
        analyze_bone_motion("Right_wrist", start_frame, end_frame)
        analyze_bone_motion("Left_wrist", start_frame, end_frame)
        bpy.ops.object.select_all(action='SELECT')  
        bpy.ops.object.delete()