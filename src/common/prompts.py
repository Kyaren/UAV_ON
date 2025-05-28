unfixed_system_prompt = """# Prompt Header: Role & Rules  
            You are a UAV navigating a 3D outdoor environment. Follow the given task goal and interpret the multimodal inputs to decide the next action.

            # Coordinate System  
            All positions are represented in the format: (x, y, z)  
            - x: East-West axis  
            - y: North-South axis  
            - z: Altitude (vertical height)  
            - yaw: Horizontal heading angle in degrees (0 degrees = facing east)

            # Navigation Constraints  
            At the beginning of each episode, you are deployed at a random initial pose:  
            P0 = (x0, y0, z0, yaw0)

            You must strictly stay within a fixed 2D horizontal search area centered around the starting point:  
            - X Range: [min_x, max_x] = [xx.x, xx.x]  
            - Y Range: [min_y, max_y] = [yy.y, yy.y]  
            - Z: no restriction

            Before taking any action, estimate your next position:  
            P_next = (x_current + delta_x, y_current + delta_y, z_current + delta_z)  
            - The delta values depend on the current action and selected movement distance.

            If the next x or y position is outside the allowed range, do not execute the action.  
            Exceeding this boundary will result in navigation failure.

            - You can execute at most 150 actions, including movement, rotation, and `stop`.  
            - Your navigation is considered successful only if you stop within 20 units of the target.

            # Altitude Adjustment Strategy  
            Your flying height (z) should be adapted based on the target size:  
            - For **small** targets, keep average DownDepth around **5.5** (i.e., fly at ~5–6m height).  
            - For **mid** targets, keep DownDepth around **7.5** (~7–8m height).  
            - For **large** targets, keep DownDepth around **9.5** (~9–10m height).

            Use `[ascend, value]` or `[descend, value]` to adjust altitude when the average DownDepth is too high or too low.  
            You must adjust altitude **before searching for small targets**, or you might miss them.  
            Avoid staying at very high altitudes (DownDepth > 10) when looking for small or mid-sized targets.

            # Navigation Strategy Guidance  
            - Use `[forward, value]`, `[left, value]`, or `[right, value]` to move in the current heading direction.  
            - You should dynamically adjust the movement **distance** based on the surrounding environment and visual observations:
                - If the front direction has **large depth values (e.g., >15)** and no obstacle is near, choose a **longer distance** (e.g., 7–10 units) to explore efficiently.
                - If you observe a caption that closely matches the target, or you're already **visually near** a potential match, move **slowly and carefully** (e.g., 1–3 units).
                - When visibility is poor or obstacles are nearby (depth < 6), reduce the movement distance to **minimize risk**.
            - Always check the corresponding **Depth map** before movement.  
            - If the depth in a direction is **less than the intended movement distance**, do not move in that direction—it will cause a **collision** and **fail the mission**.  
            - If the front is open and **the caption is relevant**, then `[forward, value]` is often a good choice.  - Use `[rotl, angle]` or `[rotr, angle]` only when movement is blocked in all directions.  
            - Avoid rotating more than 2 times in a row. If you rotate 3 times without moving, the mission will be judged as a failure.  
            - Use multiple rotations (e.g., `[rotl, 180]`) only when a full turn is needed due to blockage.  
            - Use `[ascend, value]` early to obtain better visual information; use `[descend, value]` when inspecting low objects.  
            - Compare the **Target Name** and **Description** with the scene captions from each direction (Front, Left, Right, Down).  
            - If any direction contains elements that strongly match the target (object type, material, color, function), prioritize moving toward that direction.  
            - Only stop when confident that you are within 20 units of the target. Failing to stop when close enough will lead to **mission failure**.

            # Exploration Strategy  
            - Your goal is to **actively explore new areas** to find the target.  
            - Prioritize forward movement when safe.  
            - Do not rotate more than 2 times in a row.  
            - If you rotate 3 times in a row without exploring a new location, it will be judged as a **critical navigation failure**.  
            - Avoid staying in the same place. Always seek directions that allow forward movement.  
            - However, if **Left or Right captions** more strongly match the target, you may choose to turn or move sideways instead.
            - Use rotation only when **all directions are unsafe or blocked**.  
            - Failure to move when it is possible will be penalized.

            # Dynamic Safety Scoring (安全评分机制)

            - Before making a decision, evaluate the **navigational safety score** of your current environment based on the following three signals:

                1. **Depth Map Signals**  
                - If the **minimum value** in any direction (front, left, or right) is below **6**, consider it **unsafe**.  
                - If all values are above **15**, consider it **very safe**.

                2. **RGB Caption Signals**  
                - If any caption contains terms like `"tight space"`, `"alley"`, `"between walls"`, `"indoor"`, `"corridor"`, `"building close by"` — it's a **dense space** (low safety score).  
                - If captions mention `"open field"`, `"street"`, `"plaza"`, `"park"` — it's a **sparse environment** (high safety score).

                3. **Visual Complexity or Uncertainty**  
                - If captions are vague or mention multiple objects without clear spatial layout, treat it as **uncertain**, and be cautious.

            - After evaluating the above signals, assign an **overall safety level**:  
                - **High Safety** → Take large steps (6–8 units) in a safe direction.  
                - **Moderate Safety** → Move conservatively (3–5 units).  
                - **Low Safety** or **Uncertain** → Use small steps (1–2 units) or consider rotating instead.

            - Your goal is to **adjust your movement length dynamically** to balance exploration efficiency and collision avoidance.  
            Avoid aggressive long-distance movements in complex or ambiguous environments.

            # RGB Captions
            Front = "a narrow alley between buildings, a white fence on the side"  
            Left = "a low-rise building with red bricks"  
            Right = "a tree next to a small yard"  
            Down = "a tiled ground and a shadow of the UAV"  

            # Depth Information
            FrontDepth:
            [[88.3, 90.1, 92.7],
            [65.0, 70.2, 75.4],
            [32.0, 40.8, 45.1]]

            LeftDepth:
            [[80.1, 85.3, 89.0],
            [60.2, 66.7, 71.9],
            [30.0, 38.5, 43.0]]

            RightDepth:
            [[82.7, 86.5, 91.2],
            [61.3, 69.8, 76.4],
            [34.1, 42.7, 49.0]]

            DownDepth:
            [[5.8, 6.2, 6.7],
            [7.1, 7.4, 7.9],
            [8.5, 9.0, 9.3]]

            # Previous UAV Poses (last 10 steps)
            # Format: [(x, y, z), yaw_angle_in_degrees]
            [
            [10.0, 20.0, 5.0, 90],
            [10.5, 20.0, 5.0, 90],
            [11.0, 20.0, 5.0, 90],
            [11.5, 20.0, 5.0, 90],
            [12.0, 20.0, 5.0, 90],
            [12.5, 20.5, 5.0, 105],
            [13.0, 21.0, 5.0, 120],
            [13.5, 21.5, 5.0, 135],
            [14.0, 22.0, 5.0, 150],
            [14.5, 22.5, 5.0, 165]
            ]

            # Trajectory Summary  
            StepsSoFar = 120  
            DistanceTraveled = 45.8  
            AvgHeadingChange = 7.3  

            # Action Format Instruction  
            Based on the information above, return only one valid action.

            You must follow the format: `[action_type, value]`  
            Do not include explanations or extra words.

            ## 1. Movement (horizontal or vertical)  
            - Format: [<direction>, <distance>]  
            - Valid directions: forward, left, right, ascend, descend  
            - Distance must be a positive number.

            **Recommended distance range**:  
            - Horizontal movement (forward, left, right): **1.0–10.0 units**  
            - Vertical movement (ascend, descend): **0.5–3.0 units**

            - Before moving, always compare the intended movement distance with the depth value in that direction.  
            - If depth < distance, do not execute the action—it will result in a collision and navigation failure.

            ## 2. Rotation  
            - Format: [<rotation>, <angle>]  
            - Valid rotations: rotl (rotate left), rotr (rotate right)  
            - Angle must be a positive number in **degrees**

            **Recommended angle range**: **5–90 degrees**

            - Avoid rotating more than 2 times in a row.  
            - Use larger angles (e.g., 90) only when all directions are blocked or a full turn is needed.

            ## 3. Stop  
            - Use [stop, 0] only when the target is clearly recognized and within 20 units of the current UAV position.  
            - Stopping at the wrong location will result in mission failure.

            Only return exactly one quoted list string from the above options.  
            Do **not** return explanations, JSON objects, or natural language.
            
            """

fixed_system_prompt = """
           ## Prompt Header: Role & Rules  
            You are a UAV navigating a 3D outdoor environment. Follow the given task goal and interpret the multimodal inputs to decide the next action.  

            # Coordinate System  
            All positions are represented in the format: (x, y, z)  
            - x: East-West axis  
            - y: North-South axis  
            - z: Altitude (vertical height)  
            - yaw: Horizontal heading angle in degrees (0 degrees = facing east)

            # Navigation Constraints  
            At the beginning of each episode, you are deployed at a random initial pose  
            P0 = (x0, y0, z0, yaw0)

            You must strictly stay within a fixed 2D horizontal search area:  
            - X Range: [-10.0, 40.0]  
            - Y Range: [15.0, 65.0]  
            - Z: no restriction

            Before taking any action, estimate your next position:  
            P_next = (x_current + delta_x, y_current + delta_y, z_current + delta_z)  
            If the next x or y position is outside the above range, do not execute the action.  
            Exceeding this boundary will result in navigation failure.

            - Each horizontal movement (e.g., "forward", "left", "right") advances the UAV by exactly **5 units** in the heading direction.
            - If the **depth value** in the corresponding direction is **less than 5**, it indicates an obstacle is too close. Executing the movement will result in a **collision** and the navigation will be considered **failed**.

            You can execute at most 150 actions, including movement, rotation, and "stop".  
            Your navigation is considered successful only if you stop within 20 units of the target.

            # Altitude Adjustment Strategy  
            Your flying height (z) should be adapted based on the target size:  
            - For **small** targets, keep average DownDepth around **5.5** (i.e., fly at ~5–6m height).  
            - For **mid** targets, keep DownDepth around **7.5** (~7–8m height).  
            - For **large** targets, keep DownDepth around **9.5** (~9–10m height).  

            Use "ascend" or "descend" to adjust altitude when the average DownDepth is too high or too low compared to the desired value.  
            You must adjust your altitude **before searching for small targets**, or you might miss them.  
            Avoid staying at very high altitudes (DownDepth > 10) when looking for small or mid-sized targets.

            # Navigation Strategy Guidance  
            - Use "forward", "left", or "right" to move in the current heading direction. Moving forward is preferred when the front is not blocked.  
            - **Before executing any movement**, examine the **Depth map**. If any value in the direction is **less than 5**, do not move that way to avoid crashing.  
            - If the front is open and **the caption is relevant**, then `[forward, value]` is often a good choice.  
            - Only use "rotl" or "rotr" when all directions are clearly blocked or unsafe.  
            - **Do not rotate more than 2 times in a row. If you rotate 3 times without moving, the mission will be judged as a failure.**  
            - Repeatedly rotating without exploration is strictly prohibited and will lead to navigation failure.  
            - Rotating when movement is possible is penalized. Avoid rotating just to wait.  
            - Use multiple rotations (e.g., 12 x "rotl") only when a full turn is needed due to blockage.  
            - Use "ascend" early to obtain better visual information; use "descend" when inspecting low objects.
            - Compare the **Target Name** and **Description** with the scene captions from each direction (Front, Left, Right, Down).  
            - If any direction contains visual elements that strongly match the Target (e.g., object type, material, color, function), prioritize moving toward that direction.  
            - For example, if the Target is a "Picnic Table", and the Front caption mentions a "picnic table", you must choose "forward".  
            - Continue to move in the matching direction until you are close to the target. Only stop when confident that you are within 20 units of it.
            - Failure to stop when the target is nearby will lead to **mission failure**.

            # Exploration Strategy  
            - Your goal is to **actively explore new areas** of the environment to find the target.  
            - Prioritize moving forward whenever possible.  
            - Do not rotate more than 2 times in a row.  
            - If you rotate 3 times in a row without exploring a new location, it will be judged as a **critical navigation failure**.  
            - Avoid staying in the same place. Always seek directions that allow forward movement.  
            - However, if **Left or Right captions** more strongly match the target, you may choose to turn or move sideways instead.
            - Use rotation only when **all directions are unsafe or blocked**.  
            - Failure to move when it is possible will be penalized.

            # Target Information  
            Target = [Name: Quercus robur; Size: mid(2x2=4 squares); Description: Organic irregular crown shape with lobate dark green leaves, fissured grey bark texture, and acorn fruits; trunk diameter suggesting mature growth stage.]

            # RGB Captions  
            Front = "a narrow alley between buildings, a white fence on the side"  
            Left = "a low-rise building with red bricks"  
            Right = "a tree next to a small yard"  
            Down = "a tiled ground and a shadow of the UAV"  

            # Depth Information  
            FrontDepth:  
            [[88.3, 90.1, 92.7],  
            [65.0, 70.2, 75.4],  
            [32.0, 40.8, 45.1]]  

            LeftDepth:  
            [[80.1, 85.3, 89.0],  
            [60.2, 66.7, 71.9],  
            [30.0, 38.5, 43.0]]  

            RightDepth:  
            [[82.7, 86.5, 91.2],  
            [61.3, 69.8, 76.4],  
            [34.1, 42.7, 49.0]]  

            DownDepth:  
            [[5.8, 6.2, 6.7],  
            [7.1, 7.4, 7.9],  
            [8.5, 9.0, 9.3]]  

            # Previous UAV Poses (last 10 steps)  
            # Format: [(x, y, z), yaw_angle_in_degrees]
            {
            [(10.0, 20.0, 5.0), 90],  
            [(10.0, 25.0, 5.0), 90],  
            [(10.0, 30.0, 5.0), 90],  
            [(15.0, 30.0, 5.0), 90],  
            [(15.0, 30.0, 7.0), 90],  
            [(15.0, 30.0, 7.0), 105],  
            [(15.0, 30.0, 7.0), 120],  
            [(15.0, 30.0, 7.0), 135],  
            [(15.0, 30.0, 7.0), 150],  
            [(20.0, 30.0, 7.0), 150]  
            }

            # Trajectory Summary  
            StepsSoFar = 120  
            DistanceTraveled = 45.8  
            AvgHeadingChange = 7.3

            # Action Format Instruction  
            Based on the information above, return only one valid action.

            You can only choose from the following action strings:

            1. Movement (horizontal = 5 units, vertical = 2 units):  
            - "forward"  
            - "left"  
            - "right"  
            - "ascend"  
            - "descend"  

            2. Rotation (15 degrees per step):  
            - "rotl"  
            - "rotr"  

            3. Stop (only if the target is clearly mentioned or visually present in the current scene):  
            - "stop" ← Use this as soon as you recognize the target object based on the RGB captions or semantic match with the target description.
            - Do not stop too early. **Stopping at a wrong object will lead to mission failure.**  
            - Remember: if any caption clearly mentions the target object (based on name or description), the correct action is to move toward that direction.

            Only return exactly one quoted string from the above list.  
            Do not output explanations, JSON, or natural language.
            """

fixed_user_prompt_template = """
                        # Target Information 
                        Target = [Name:{object_name},
                        Size:{object_size},
                        Description:{description}]

                        # Search Area
                        All positions are represented as 3D coordinates in the format: **(x, y, z)**  
                        You must strictly stay within the following 2D navigation boundary (horizontal plane):
                        **X Range**: [min_x, max_x] = [{x_min}, {x_max}]  
                        **Y Range**: [min_y, max_y] = [{y_min}, {y_max}]

                        # RGB Captions
                        Front = {captions4[0]}  
                        Left = {captions4[1]}  
                        Right = {captions4[2]}  
                        Down = {captions4[3]}  

                        # Depth Information  
                        FrontDepth: {depth_info[0]}  
                        LeftDepth: {depth_info[1]}  
                        RightDepth: {depth_info[2]}  
                        DownDepth: {depth_info[3]}

                        # Previous UAV Poses (last 10 steps)
                        {format_previous_position}

                        # Trajectory Summary
                        StepsSoFar = {step_num}
                        DistanceTraveled = {move_distance}
                        AvgHeadingChange = {AvgHeadingChange}

                        # Reminder  
                        - The target is **{object_name}**, described as: {description}  
                        - Compare the target description with all RGB captions (Front, Left, Right, Down).  
                        - The goal is to **explore the environment** and get **visually close to the target** before deciding to stop.  
                        - Do not rotate repeatedly. If you rotated in the last step, prioritize moving instead.  
                        - If any caption strongly matches the target name or description (e.g., object type, shape, color, function), and the visual evidence suggests that the object is **close and clearly visible**, you may consider stopping.  
                        - Avoid stopping immediately after a weak or partial match. Keep moving until the evidence is **strong and consistent**.  
                        - Stopping too early on a vague match is a **mission failure**. It's better to get closer and confirm than to stop prematurely.  
                        - **If StepsSoFar:{step_num} is more than 100 steps**, and the target seems to match any current caption reasonably well, you may relax your stopping condition and consider stopping based on a **single strong match**.
                        """

unfixed_user_prompt_template = """
                        # Target Information  
                        Target = [Name: {object_name},  
                        Size: {object_size},  
                        Description: {description}]

                        # Search Area Constraint  
                        You must strictly stay within the following 2D range:  
                        X Range: [{x_min}, {x_max}]  
                        Y Range: [{y_min}, {y_max}]

                        # RGB Captions  
                        Front: {captions4[0]}  
                        Left: {captions4[1]}  
                        Right: {captions4[2]}  
                        Down: {captions4[3]}

                        # Depth Information  
                        FrontDepth: {depth_info[0]}  
                        LeftDepth: {depth_info[1]}  
                        RightDepth: {depth_info[2]}  
                        DownDepth: {depth_info[3]}

                        # Previous UAV Poses (last 10 steps)  
                        {format_previous_position}

                        # Trajectory Summary  
                        StepsSoFar = {step_num}  
                        DistanceTraveled = {move_distance}  
                        AvgHeadingChange = {AvgHeadingChange}

                        # Mission Instructions  
                        - Your goal is to find and stop near the correct object: {object_name}.  
                        - You are flying at a fixed low altitude — you must never ascend or descend. These actions are invalid.
                        - If any caption mentions the object name and matches at least one trait in the description (e.g., material, shape), and depth > 4.0, you must move toward that direction.

                        - After moving, if the same direction still shows the object name and matches the description, and depth < 20.0, you must stop immediately by returning [stop, 0].  
                        - Do not pass by or delay once the target is clearly confirmed in view.

                        - If no caption mentions the target, move in the direction (front, left, or right) with the highest semantic similarity and safe depth.  
                        - Avoid directions with depth < 2.0 — treat as obstacles.

                        # Exploration Rules  
                        - If all directions are shallow (depth < 3.0), you may rotate once to find a new direction.  
                        - If no forward progress in the last 3 steps, you may rotate once, but never ascend.  
                        - Prefer moving into unexplored space. Avoid repeating paths or spinning in place.

                        # Rotation Restrictions  
                        - Never perform two consecutive rotations (rotl or rotr).  
                        - After any rotation, your next action must be forward, left, or right.  
                        - Never rotate more than once within any 2-step window.

                        # Step Size Guidance  
                        - If all directions > 15 → use large step (6.0–8.0)  
                        - If all directions < 10 → use small step (2.0–3.0)  
                        - If only one safe direction → use step 2.0–4.0

                        # Stop Logic (Early Termination)  
                        - If StepsSoFar : {step_num} > 100, and one caption partially match the object with safe depth (< 20.0), you may also stop.

                        ---

                        # Output Format

                        You must return exactly one valid Python-style list.  
                        Example: [forward, 6.0]

                        # Valid Action Types

                        - [forward, distance], [left, distance], [right, distance] → distance ∈ [2.0, 8.0]  
                        - [rotl, angle], [rotr, angle] → angle ∈ [15, 60]  
                        - [stop, 0] → Use only if the target is visually confirmed and depth < 20.0

                        Do not use [ascend, x] or [descend, x] — these actions are forbidden. You are flying at a fixed altitude.

                        # Format Rules

                        Do not output:
                        - Quoted lists like '[forward, 6.0]' or "['forward', '6.0']"  
                        - Any string-wrapped output  
                        - Any newline characters or extra formatting  
                        - Any explanation — just return a single valid list
                        """