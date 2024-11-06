import os

# Template for the .sumocfg file content
sumocfg_template = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="3_2_merge.net.xml"/>
        <route-files value="generated_flows_{model}_{index}.rou.xml"/>
        <additional-files value="loops_detectors.xml"/>
        <gui-settings-file value="colored.view.xml"/>
    </input>
</configuration>
"""

def create_sumocfg(model, num_envs_per_model):
     output_dir = "./traffic_environment/sumo"
     os.makedirs(output_dir, exist_ok=True)

     # Generate configuration files
     for i in range(num_envs_per_model):
          # Create filename based on model and environment index
          filename = f"3_2_merge_{model}_{i}.sumocfg"
          filepath = os.path.join(output_dir, filename)
          
          # Format the template with current model and index
          content = sumocfg_template.format(model=model, index=i)
          
          # Write the content to the file
          with open(filepath, 'w') as file:
               file.write(content)
          
          print(f"Created {filepath}")