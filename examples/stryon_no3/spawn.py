import os
import genesis as gs

gs.init(backend=gs.cuda)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "../../genesis/assets/xml/stryon_no3/stryon_no3.xml")
stryon_no3 = scene.add_entity(gs.morphs.MJCF(file=xml_path))

scene.build()

while True:
    scene.step()
