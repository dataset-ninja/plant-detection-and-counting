The authors of **Plant Detection and Counting** dataset emphasize the importance of accurately measuring agronomic traits, particularly plant density at emergence, which plays a crucial role in crop yield for plants like maize, sugar beet, and sunflower. They highlight that this trait is influenced by seeding density, emergence rate, and planting pattern, impacting plant competition and weed growth. Furthermore, the authors note that plant density assessment is valuable in both research experiments and farming decisions, despite being indirectly influenced by factors like seeding density, seed vigor, and emergence conditions.

The authors utilized a dataset from maize, sugar beet, and sunflower experiments conducted between 2016 and 2019 at diverse experimental sites in France. These sites had varying agronomic conditions, conventional tillage practices, and soil types, including clay, brunisolic, and limestone, affecting surface characteristics and soil color. Each site had multiple microplots representing different genotypes, with 3 to 12 microplots selected to yield around 600 plants. Some sites were surveyed multiple times, providing a broader range of crop development stages during image acquisition. For maize, the authors had 51 microplots from 9 acquisition sessions with varying microplot size, row spacing (0.3-1.1m), and plant density (5.1-11.2 plt.m-2). Sugar beet had 60 microplots from 9 sessions with minor variations in microplot attributes, while sunflower featured 78 microplots from 9 sessions with substantial variability in microplot characteristics.

The authors employed Unmanned Aerial Vehicles (UAVs) equipped with three different RGB cameras: the Sony Alpha 5100 and Sony Alpha 6000, both with a resolution of 6024×4024 pixels, and the Zenmuse X7 (DJI) for the Epoisses site in 2019, with a resolution of 6016 x 4008 pixels. These cameras were stabilized on a two-axis gimbal to maintain a nadir view during flight and were set to a speed priority of 1/1250 s to reduce motion blur. Aperture and ISO settings were automatically adjusted, and the cameras were triggered at a 1Hz interval to capture RGB images in JPG format. Flight altitudes varied between 20 to 50 meters to achieve a ground sampling distance (GSD) ranging from 2 mm to 5 mm per pixel (Table 3). The flight trajectory was planned to ensure more than 70% overlap between images along both lateral and longitudinal tracks. Ground control points were strategically placed in the field and their coordinates were measured using real-time kinematic GPS, ensuring centimeter-level accuracy in positioning.

The authors utilized Agisoft Photoscan Professional software for image alignment and employed a structure from motion algorithm to determine camera positions and orientations. They followed a pipeline described by Jin et al. to extract microplot portions from each image, using a georeferenced plot map to avoid distortions seen in orthomosaics. The sharpest extract containing the entire microplot was selected for labeling, with around 600 plants labeled per session, resulting in a total of 16,247 labeled plants. Images were rescaled to a consistent GSD of 2.5 mm, and labeling was performed using the coco-annotator tool, involving six different operators with a review process. The plant development stage, weed infestation level, and image blurriness were also assessed for each session.