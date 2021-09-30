from scenegenerator import Camera, Object, Material, SceneFile

sceneFile = SceneFile()
sceneFile.addMaterial(Material(
    tag='// light',
    RGB=[1, 1, 1],
    EMITTANCE=10))
sceneFile.addMaterial(Material(
    tag='// Diffuse white',
    RGB=[0.98, 0.98, 0.98]))
sceneFile.addMaterial(Material(
    tag='// Diffuse red',
    RGB=[0.85, 0.35, 0.35]))
sceneFile.addMaterial(Material(
    tag='// Diffuse green',
    RGB=[0.35, 0.85, 0.35])) 
sceneFile.addMaterial(Material(
    tag='// Diffuse cyan',
    RGB=[0, 0.98, 0.98]))      
sceneFile.addMaterial(Material(
    tag='// Diffuse fuchsia',
    RGB=[0.98, 0, 0.98]))                                                                
sceneFile.addMaterial(Material(
    tag='// Specular white',
    RGB=[0.98, 0.98, 0.98],
    SPECRGB=[0.98, 0.98, 0.98],
    REFL=1))
sceneFile.addMaterial(Material(
    tag='// Refractive white',
    RGB=[0.98, 0.98, 0.98],
    SPECRGB=[0.98, 0.98, 0.98],
    REFR=1,
    REFRIOR=1.5))

sceneFile.addCamera(Camera(
    DEPTH=8,
    RES=[800, 800],
    FOVY=45,
    ITERATIONS=4000))

sceneFile.addObject(Object(
    tag='// Ceiling light L',
    flavor='cube',
    material='// light',
    TRANS=[-3, 10, 0],
    SCALE=[2, 0.3, 2]))
sceneFile.addObject(Object(
    tag='// Ceiling light R',
    flavor='cube',
    material='// light',
    TRANS=[3, 10, 0],
    SCALE=[2, 0.3, 2]))
sceneFile.addObject(Object(
    tag='// Floor',
    flavor='cube',
    material='// Diffuse white',
    SCALE=[10, 0.01, 10]))
sceneFile.addObject(Object(
    tag='// Ceiling',
    flavor='cube',
    material='// Diffuse white',
    TRANS=[0, 10, 0],
    ROTAT=[0, 0, 90],
    SCALE=[0.01, 10, 10]))
sceneFile.addObject(Object(
    tag='// Back wall',
    flavor='cube',
    material='// Diffuse white',
    TRANS=[0, 5, -5],
    ROTAT=[0, 90, 0],
    SCALE=[0.01, 10, 10]))
sceneFile.addObject(Object(
    tag='// Left wall',
    flavor='cube',
    material='// Diffuse red',
    TRANS=[-5, 5, 0],
    SCALE=[0.01, 10, 10]))
sceneFile.addObject(Object(
    tag='// Right wall',
    flavor='cube',
    material='// Diffuse green',
    TRANS=[5, 5, 0],
    SCALE=[0.01, 10, 10]))

sceneFile.addObject(Object(
    tag='// L sphere',
    flavor='sphere',
    material='// Specular white',
    TRANS=[-3.5, 2, 2],
    SCALE=[3, 3, 3]))

sceneFile.addObject(Object(
    tag='// C sphere',
    flavor='sphere',
    material='// Refractive white',
    TRANS=[0, 2, 2],
    SCALE=[3, 3, 3]))

sceneFile.addObject(Object(
    tag='// R sphere',
    flavor='sphere',
    material='// Diffuse fuchsia',
    TRANS=[3.5, 2, 2],
    SCALE=[3, 3, 3]))

# sceneFile.addObject(Object(
#     tag='// Cow',
#     flavor='cow.obj',
#     material='// Diffuse cyan',
#     TRANS=[-1, 2.7, 0],
#     ROTAT=[0, 0, 0],
#     SCALE=[0.75, 0.75, 0.75]))

sceneFile.generate('spheres')