from scenegenerator import Camera, Object, Material, SceneFile

sceneFile = SceneFile()
sceneFile.addMaterial(Material(
    tag='// light',
    RGB=[1, 1, 1],
    EMITTANCE=50))
sceneFile.addMaterial(Material(
    tag='// cow light',
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
    RES=[1200, 800],
    FOVY=45,
    ITERATIONS=3000))

# sceneFile.addObject(Object(
#     tag='// Ceiling light L',
#     flavor='sphere',
#     material='// light',
#     TRANS=[10, 50, 0],
#     SCALE=[1, 1, 1]))
# sceneFile.addObject(Object(
#     tag='// Ceiling light R',
#     flavor='sphere',
#     material='// light',
#     TRANS=[-10, 50, 0],
#     SCALE=[1, 1, 1]))
sceneFile.addObject(Object(
    tag='// Floor',
    flavor='cube',
    material='// Diffuse green',
    SCALE=[100, 0.01, 100]))
sceneFile.addObject(Object(
    tag='// Ceiling',
    flavor='cube',
    material='// Diffuse white',
    TRANS=[0, 100, 0],
    ROTAT=[0, 0, 90],
    SCALE=[0.01, 10, 10]))

sceneFile.addObject(Object(
    tag='// Cow',
    flavor='cow.obj',
    material='// Diffuse fuchsia',
    TRANS=[-3, 2.7, 2],
    ROTAT=[0, -25, 0],
    SCALE=[0.75, 0.75, 0.75]))

sceneFile.addObject(Object(
    tag='// Cow',
    flavor='cow.obj',
    material='// Diffuse fuchsia',
    TRANS=[-1, 2.7, -4],
    ROTAT=[0, 25, 0],
    SCALE=[0.75, 0.75, 0.75]))

sceneFile.addObject(Object(
    tag='// Cow',
    flavor='cow.obj',
    material='// Diffuse fuchsia',
    TRANS=[3, 2.7, -1],
    ROTAT=[0, 0, 0],
    SCALE=[0.75, 0.75, 0.75]))

sceneFile.addObject(Object(
    tag='// Cow',
    flavor='cow.obj',
    material='// Diffuse fuchsia',
    TRANS=[4, 2.7, 6],
    ROTAT=[0, 30, 0],
    SCALE=[0.75, 0.75, 0.75]))

sceneFile.addObject(Object(
    tag='// Cow',
    flavor='cow.obj',
    material='// cow light',
    TRANS=[3, 10, 4],
    ROTAT=[0, 0, -120],
    SCALE=[0.75, 0.75, 0.75]))

sceneFile.addObject(Object(
    tag='// Ufo',
    flavor='ufo.obj',
    material='// Specular white',
    TRANS=[4, 14, 4],
    ROTAT=[0, 0, -20],
    SCALE=[0.75, 0.75, 0.75]))


# sceneFile.addObject(Object(
#     tag='// Cow',
#     flavor='cow.obj',
#     material='// Diffuse cyan',
#     TRANS=[0, 2.7, 0],
#     ROTAT=[0, 0, 0],
#     SCALE=[0.75, 0.75, 0.75]))


sceneFile.generate('epic')