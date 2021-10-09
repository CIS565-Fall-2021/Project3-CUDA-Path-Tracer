spread_array = lambda array: ' '.join([str(e) for e in array])

class Material(object):
    def __init__(self, tag='// asdf', RGB=[1, 0, 0], SPECEX=0, SPECRGB=[0, 0, 0], REFL=0, REFR=0, REFRIOR=0, EMITTANCE=0):
        self.tag = tag
        self.rgb = RGB
        self.specex = SPECEX
        self.specrgb = SPECRGB
        self.refl = REFL
        self.refr = REFR
        self.refrior = REFRIOR
        self.emittance = EMITTANCE

    def get_lines(self, i):
        return [
            self.tag,
            f'MATERIAL {i}',
            f'RGB\t\t{spread_array(self.rgb)}',
            f'SPECEX\t\t{self.specex}',
            f'SPECRGB\t\t{spread_array(self.specrgb)}',
            f'REFL\t\t{self.refl}',
            f'REFR\t\t{self.refr}',
            f'REFRIOR\t\t{self.refrior}',
            f'EMITTANCE\t\t{self.emittance}'
        ]

class Object(object):
    def __init__(self, tag='// asdf', flavor='sphere', material=0, TRANS=[0, 0, 0], ROTAT=[0, 0, 0], SCALE=[1, 1, 1]):
        self.tag = tag
        self.flavor = flavor
        self.material = material
        self.trans = TRANS
        self.rotat = ROTAT
        self.scale = SCALE

    def get_lines(self, i):
        return [
            self.tag,
            f'OBJECT {i}',
            self.flavor,
            f'material {self.material}',
            f'TRANS\t\t{spread_array(self.trans)}',
            f'ROTAT\t\t{spread_array(self.rotat)}',
            f'SCALE\t\t{spread_array(self.scale)}',
        ]

class Camera(object):
    def __init__(self, tag='// Camera', RES=[800, 800], FOVY=45, ITERATIONS=5000, DEPTH=6, FILE='cornell', EYE=[0, 5, 10.5], LOOKAT=[0, 5, 0], UP=[0, 1, 0]):
        self.tag = tag
        self.res = RES
        self.fovy = FOVY
        self.iterations = ITERATIONS
        self.depth = DEPTH
        self.file = FILE
        self.eye = EYE
        self.lookat = LOOKAT
        self.up = UP

    def get_lines(self):
        return [
            self.tag,
            'CAMERA',
            f'RES\t\t{spread_array(self.res)}',
            f'FOVY\t\t{self.fovy}',
            f'ITERATIONS\t\t{self.iterations}',
            f'DEPTH\t\t{self.depth}',
            f'FILE\t\t{self.file}',
            f'EYE\t\t{spread_array(self.eye)}',
            f'LOOKAT\t\t{spread_array(self.lookat)}',
            f'UP\t\t{spread_array(self.up)}'
        ]

class SceneFile(object):
    def __init__(self):
        self.objects = []
        self.materials = []

    def addCamera(self, camera):
        self.camera = camera

    def addObject(self, object):
        self.objects.append(object)

    def addMaterial(self, material):
        self.materials.append(material)

    def generate(self, filename):
        f = open(f"{filename}.txt", "w")
        
        matTag2Num = {}
        # MATERIALS
        for i, material in enumerate(self.materials):
            for j, line in enumerate(material.get_lines(i)):
                if j == 0:
                    matTag2Num[line] = i
                f.write(line + '\n')
            f.write('\n')

        # CAMERA
        for line in self.camera.get_lines():
            f.write(line + '\n')
        f.write('\n')

        # OBJECTS
        for i, object in enumerate(self.objects):
            # convert string label to number
            object.material = matTag2Num[object.material]
            for j, line in enumerate(object.get_lines(i)):
                f.write(line + '\n')
            f.write('\n')

        f.close()

