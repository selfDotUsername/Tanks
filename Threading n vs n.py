import pygame
import numpy as np
import copy
import pickle
import time
import threading
import os.path

# display and scale
bulletsSize = [25, 25]
tanksSize = [50, 50]
window_width = 1000
window_height = 1000
fps = 30

# graphics
tanksImages = ['tank_chassis .png', 'tank_turret.png']
bulletsImage = 'bullet_tail.png'

# movement
##tanks
tanksVelocity = 5
tanksRotation = 0.1
turretRotation = 0.1
##bullets
bulletsVelocity = 7

# vision
lidarSize = 10
nBodyLidar = 4
nTurretLidar = 1

# model
nRecurrence = 4
networkShape = [10]

# timings
tanksShotFrames = 100
framesToUpdatePos = 1000
staleReset = 1

# game
nModels = 9
tanksPerRound = 1
killsForWin = 1

# save/load
keepStats = 1
loadFrom = "Instances02"
saveTo = "Instances02"
training = 1#1 for training 0 for evaluation


#evolution
evolutionRate = 1.5

pygame.init()
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Tanks")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)


def update_fps():
    fps = str(int(clock.get_fps()))
    fps_text = font.render("fps:" + fps, 1, pygame.Color("coral"))
    return fps_text


def rot_center(image, angle):
    """rotate an image while keeping its center and size"""
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image


def isPointAhead(x0, y0, x1, y1, c, s):
    return c * (x1 - x0) + s * (y1 - y0) > 0


def squaredistance(x0, y0, x1, y1):
    return (x0 - x1) ** 2 + (y0 - y1) ** 2


def isLidarCollision(X0, X1, dmax, sin, cos):
    x0, y0, x1, y1 = X0[0], X0[1], X1[0], X1[1]
    c = cos
    s = sin
    termsInCommon = (x1 - x0) * c + (y1 - y0) * s
    xbar = x0 + c * termsInCommon
    ybar = y0 + s * termsInCommon
    d2 = squaredistance(xbar, ybar, x1, y1)
    if d2 <= dmax and isPointAhead(x0, y0, xbar, ybar, c, s):
        l2 = squaredistance(xbar, ybar, x0, y0)
        return l2
    return None


def isL1Collisoin(X0, X1, d):
    return np.abs(X0[0] - X1[0]) < d and np.abs(X0[1] - X1[1]) < d


def sigmoid(_x):
    return 1 / (1 * np.e ** (-_x))


def random(r0, r1):
    return np.random.rand() * (r1 - r0) + r0


def zeros(n):
    answer = []
    for i in range(n):
        answer.append(0)
    return answer


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Network:
    def __init__(self, shape, activation):
        self.layers = []
        self.nlayers = len(shape) - 1
        self.activation = activation()
        for i in range(self.nlayers):
            self.layers.append(Layer_Dense(shape[i], shape[i + 1]))

    def forward(self, inputs):
        self.layers[0].forward(inputs)
        self.activation.forward(self.layers[0].output)
        for i in range(self.nlayers - 1):
            self.layers[i + 1].forward(self.activation.output)
            self.activation.forward(self.layers[i + 1].output)
        self.output = self.activation.output
        # for i in range(len(self.output)):
        #    for j in range(len(self.output[i])):
        #        if self.output[i][j] > 0:
        #            self.output[i][j] = 1

    def child(self, rate):
        for i in range(self.nlayers):
            for j in range(len(self.layers[i].weights)):
                for k in range(len(self.layers[i].weights[j])):
                    if np.random.rand() > sigmoid(rate):
                        self.layers[i].weights[j][k] *= random(-2, 2)
                        if self.layers[i].weights[j][k] > 1000:
                            self.layers[i].weights[j][k] = 1000
                        if self.layers[i].weights[j][k] < -1000:
                            self.layers[i].weights[j][k] = -1000
            for j in range(len(self.layers[i].biases)):
                for k in range(len(self.layers[i].biases[j])):
                    if np.random.rand() > sigmoid(rate):
                        self.layers[i].biases[j][k] *= random(-2, 2)
                        if self.layers[i].biases[j][k] > 1000:
                            self.layers[i].biases[j][k] = 1000
                        if self.layers[i].biases[j][k] < -1000:
                            self.layers[i].biases[j][k] = -1000





class Tank:

    def __init__(self, instance, team, position, direction, tDirection, model):
        # args
        self.instance = instance
        self.team = team
        self.X = position
        self.d = direction
        self.td = tDirection
        self.model = model

        # movement
        self.v = tanksVelocity
        self.dv = tanksRotation
        self.tdv = turretRotation

        self.cosd = np.cos(self.d)
        self.sind = np.sin(self.d)

        self.vcosd = self.v * self.cosd
        self.vsind = self.v * self.sind

        self.costd = np.cos(self.td)
        self.sintd = np.sin(self.td)

        self.mUp = 0
        self.mDown = 0
        self.mLeft = 0
        self.mRight = 0
        self.mtLeft = 0
        self.mtRight = 0

        # lidar
        self.lidarSize = lidarSize
        self.lidarSize2 = lidarSize ** 2
        self.nBodyLidar = nBodyLidar
        self.nTurretLidar = nTurretLidar

        # model
        self.recurrence = zeros(nRecurrence)

        # instance
        self.n = len(self.instance.tanks)
        #self.instance.tanks.append(self)


        # game
        self.alive = 1
        self.shotTime = tanksShotFrames

    def up(self):
        self.X[0] += self.vcosd
        self.X[1] += self.vsind

    def down(self):
        self.X[0] -= self.vcosd
        self.X[1] -= self.vsind

    def left(self):
        self.d -= self.dv
        self.d = self.d % (2 * np.pi)
        self.vsind = self.v * np.sin(self.d)
        self.vcosd = self.v * np.cos(self.d)

    def right(self):
        self.d += self.dv
        self.d = self.d % (2 * np.pi)
        self.vsind = self.v * np.sin(self.d)
        self.vcosd = self.v * np.cos(self.d)

    def tLeft(self):
        self.td -= self.tdv
        self.td = self.td % (2 * np.pi)
        self.sintd = np.sin(self.d)
        self.costd = np.cos(self.d)

    def tRight(self):
        self.td += self.tdv
        self.td = self.td % (2 * np.pi)
        self.sintd = np.sin(self.d)
        self.costd = np.cos(self.d)

    def move(self):
        if self.mUp == 1:
            self.up()
        if self.mDown == 1:
            self.down()
        if self.mLeft == 1:
            self.left()
        if self.mRight == 1:
            self.right()
        if self.mtLeft == 1:
            self.tLeft()
        if self.mtRight == 1:
            self.tRight()
        if self.X[0] > window_width - tanksSize[0] / 2:
            self.X[0] = window_width - tanksSize[0] / 2
        if self.X[0] < tanksSize[0] / 2:
            self.X[0] = tanksSize[0] / 2
        if self.X[1] > window_height - tanksSize[1] / 2:
            self.X[1] = window_height - tanksSize[1] / 2
        if self.X[1] < tanksSize[1] / 2:
            self.X[1] = tanksSize[1] / 2

    def shoot(self):
        if self.shotTime < 1:
            Bullet(self)
            self.shotTime = tanksShotFrames

    def lidarWalls(self, d):
        _x, _y = self.X[0], self.X[1]
        if d == 0:
            # right
            # print(1)
            return [(window_width - _x) ** 2]
        if d == np.pi / 2:
            # bottom
            # print(2)
            return [_y ** 2]
        if d == np.pi:
            # left
            # print(3)
            return [_x ** 2]
        if d == 3 * np.pi / 2:
            # top
            # print(4)
            return [(window_height - _y) ** 2]

        _m = np.tan(d)

        if d < np.pi / 2:
            if _m * (window_width - _x) + _y < window_height:
                # right
                # print(5)
                return [squaredistance(_x, _y, window_width, _m * (window_width - _x) + _y)]
            else:
                # bottom
                # print(6)
                return [squaredistance(_x, _y, _x + (window_height - _y) / _m, window_height)]
        if d < np.pi:
            if _y - _m * _x <= window_height:
                # left
                # print(7)
                return [squaredistance(_x, _y, 0, _y - _m * _x)]
            else:
                # bottom
                # print(8)
                return [squaredistance(_x, _y, _x + (window_height - _y) / _m, window_height)]
        if d < 3 * np.pi / 2:
            if _y - _m * _x >= 0:
                # left
                # print(9)
                return [squaredistance(_x, _y, 0, _y - _m * _x)]
            else:
                # top
                # print(10)
                return [squaredistance(_x, _y, _x - _y / _m, 0)]
        if _x - _y / _m > window_width:
            # right
            # print(11)
            return [squaredistance(_x, _y, window_width, _m * (window_width - _x) + _y)]
        else:
            # top
            # print(12)
            return [squaredistance(_x, _y, _x - _y / _m, 0)]

    def lidarSingle(self, d):
        # print("d=",d)
        _lidarOutput = []
        _tDetections = []
        _sin = np.sin(d)
        _cos = np.cos(d)
        for i in range(len(self.instance.tanks)):
            if self.instance.tanks[i] is self:
                _tDetections.append(None)
            else:
                _tDetections.append(isLidarCollision(self.X, self.instance.tanks[i].X, self.lidarSize2, _sin, _cos))
        _bDetections = []
        for i in range(len(self.instance.bullets)):
            _bDetections.append(isLidarCollision(self.X, self.instance.bullets[i].X, self.lidarSize2, _sin, _cos))
        _tmin = None
        _tmini = None
        for i in range(len(_tDetections)):
            if _tDetections[i] is not None:
                if _tmin is None:
                    _tmin = _tDetections[i]
                    _tmini = i

                if _tmin > _tDetections[i]:
                    _tmin = _tDetections[i]
                    _tmini = i
        _bmin = None
        _bmini = None
        for i in range(len(_bDetections)):
            if _bDetections[i] is not None:
                if _bmin is None:
                    _bmin = _bDetections[i]
                    _bmini = i

                if _bmin > _bDetections[i]:
                    _bmin = _bDetections[i]
                    _bmini = i
        if _tmin is None or _bmin is None:
            _lidarWalls = self.lidarWalls(d)
            if _tmin is None and _bmin is None:
                return _lidarWalls + [10, 10, 10] + _lidarWalls + [10]
            if _bmin is None:
                return [_tmin, self.instance.tanks[_tmini].td - d, self.instance.tanks[_tmini].d - d,
                        self.instance.tanks[_tmini].team] + _lidarWalls + [10]
            if _tmin is None:
                return _lidarWalls + [10, 10, 10] + [_bmin, self.instance.bullets[_bmini].d - d]
        return [_tmin, self.instance.tanks[_tmini].td - d, self.instance.tanks[_tmini].d - d,
                self.instance.tanks[_tmini].team] + [_bmin,
                                                     self.instance.bullets[
                                                         _bmini].d - d]

    def lidarAll(self):
        _lidarList = []
        _directionStep = (2 * np.pi) / self.nBodyLidar
        for i in range(self.nBodyLidar):
            _singleLidar = self.lidarSingle((self.d + i * _directionStep) % (2 * np.pi))
            _lidarList += _singleLidar
        _directionStep = (2 * np.pi) / self.nTurretLidar
        for i in range(self.nTurretLidar):
            _singleLidar = self.lidarSingle((self.td + i * _directionStep) % (2 * np.pi))
            _lidarList += _singleLidar
        return _lidarList

    def collectAiData(self):
        # print(self.lidarAll() + [self.d, + self.td, self.shotTime] + self.recurrence)
        return self.lidarAll() + [self.d, + self.td, self.shotTime, self.team] + self.recurrence

    def forwardModel(self):
        _networkInput = self.collectAiData()
        self.model.forward([_networkInput])
        _modelOutput = self.model.output[0]

        # print(_modelOutput[0])

        if _modelOutput[0] > 0:
            self.mUp = 1
        else:
            self.mUp = 0

        if _modelOutput[1] > 0:
            self.mDown = 1
        else:
            self.mDown = 0

        if _modelOutput[2] > 0:
            self.mLeft = 1
        else:
            self.mLeft = 0

        if _modelOutput[3] > 0:
            self.mRight = 1
        else:
            self.mRight = 0

        if _modelOutput[4] > 0:
            self.mtLeft = 1
        else:
            self.mtLeft = 0

        if _modelOutput[5] > 0:
            self.mtRight = 1
        else:
            self.mtRight = 0

        if _modelOutput[6] > 0:
            self.shoot()

        for i in range(nRecurrence):
            self.recurrence[i] = (_modelOutput[7 + i] % 100)

    def delete(self):
        if self.alive == 0:
            for i in range(self.n + 1, len(self.instance.tanks)):
                self.instance.tanks[i].n -= 1
            #print(self.instance.tanks)
            #print(self.n)
            #print(self)
            self.instance.tanks.pop(self.n)
            self.instance.teams[self.team] -= 1

    def frame0(self):
        if self.shotTime != 0:
            self.shotTime -= 1
        self.forwardModel()

    def frame1(self):
        self.move()


class Bullet:

    def __init__(self, tank):
        self.tank = tank
        self.n = len(self.tank.instance.bullets)
        self.tank.instance.bullets.append(self)
        self.alive = 1
        # position
        self.X = copy.deepcopy(tank.X)
        self.v = bulletsVelocity
        self.d = copy.deepcopy(tank.td)

        self.vcosd = self.v * np.cos(self.d)
        self.vsind = self.v * np.sin(self.d)

    def move(self):
        self.X[0] += self.vcosd
        self.X[1] += self.vsind

    def delete(self):
        if self.alive == 0:
            for i in range(self.n + 1, len(self.tank.instance.bullets)):
                self.tank.instance.bullets[i].n -= 1
            self.tank.instance.bullets.pop(self.n)

    def collisionDetection(self):
        for i in range(len(self.tank.instance.tanks)):
            if self.tank.instance.tanks[i] is not self.tank and isL1Collisoin(self.X, self.tank.instance.tanks[i].X, (
                                                                                                                             bulletsSize[
                                                                                                                                 0] +
                                                                                                                             tanksSize[
                                                                                                                                 0]) / 2) and \
                    self.tank.instance.tanks[i].alive == 1:
                #print("collision")
                self.alive = 0
                self.tank.instance.tanks[i].alive = 0
        if self.X[0] > window_width or self.X[0] < 0 or self.X[1] > window_height or self.X[1] < 0:
            self.alive = 0


class Instance:
    instances = []
    nInstances = 0

    def __init__(self, model0, model1, tanksPerTeam, models, i, k):
        self.tanks = []
        self.bullets = []
        self.teams = [0, 0]

        self.models = models
        _model0 = copy.deepcopy(model0)
        _model1 = copy.deepcopy(model1)
        self.model0dir = i
        self.model1dir = k
        for i in range(tanksPerTeam):


            self.teams[0] += 1
            self.teams[1] += 1
            self.tanks.append(Tank(self, 0, [random(0, window_width), random(0, window_height)], random(0, 2 * np.pi),
                                   random(0, 2 * np.pi), _model0))
            self.tanks.append(Tank(self, 1, [random(0, window_width), random(0, window_height)], random(0, 2 * np.pi),
                                   random(0, 2 * np.pi), _model1))

    def updateTanks(self):
        for i in range(len(self.tanks)):
            self.tanks[i].frame0()
        for i in range(len(self.tanks)):
            self.tanks[i].frame1()
        _deletions = 0
        for i in range(len(self.tanks)):
            if self.tanks[i - _deletions].alive == 0:
                #print("tank",self.tanks[i - _deletions])
                #print("alive",self.tanks[i - _deletions].alive)
                #print("i=",i,"i-deletions",i-_deletions)
                self.tanks[i - _deletions].delete()
                _deletions += 1

    def updateBullets(self):
        for i in range(len(self.bullets)):
            self.bullets[i].move()
            self.bullets[i].collisionDetection()


    def drawAll(self):
        screen.fill((0,0,0))
        for i in range(len(self.tanks)):

            screen.blit(rot_center(pygame.transform.scale(pygame.image.load(tanksImages[0]), tanksSize), -(self.tanks[i].d/(2*np.pi)*360) - 90),
                        (self.tanks[i].X[0] - tanksSize[0] / 2, self.tanks[i].X[1] - tanksSize[1] / 2))

            screen.blit(rot_center(pygame.transform.scale(pygame.image.load(tanksImages[1]), tanksSize), -(self.tanks[i].td/(2*np.pi)*360) - 90),
                        (self.tanks[i].X[0] - tanksSize[0] / 2, self.tanks[i].X[1] - tanksSize[1] / 2))

            # pygame.draw.circle(screen, (255,0,0), (self.X[0] + 10*np.cos(self.td), self.X[1] + 10*np.sin(self.td)), 10)
            if self.tanks[i].team == 1:
                pygame.draw.circle(screen, (255, 0, 0), (self.tanks[i].X[0], self.tanks[i].X[1]), 2)
            else:
                pygame.draw.circle(screen, (0, 0, 255), (self.tanks[i].X[0], self.tanks[i].X[1]), 2)

        for i in range(len(self.bullets)):
            screen.blit(rot_center(pygame.transform.scale(pygame.image.load(bulletsImage), bulletsSize), -(self.bullets[i].d/(2*np.pi)*360) - 90),
                        (self.bullets[i].X[0] - bulletsSize[0] / 2, self.bullets[i].X[1] - bulletsSize[1] / 2))
        pygame.display.update()
    def frame(self):
        self.updateTanks()
        self.updateBullets()
        if training == 0:
            self.drawAll()

    def evaluate(self):

        _updatePosTimer = framesToUpdatePos
        _updatePosCount = 0
        _roundRunning = True
        while _roundRunning:
            if training == 0:
                clock.tick(fps)
            _updatePosTimer -= 1
            if _updatePosTimer < 1:
                for i in range(len(self.tanks)):
                    self.tanks[i].X = [random(0, window_width), random(0, window_height)]
                _updatePosTimer = framesToUpdatePos
                _updatePosCount += 1
                if _updatePosCount == staleReset:
                    print("draw")
                    return [self.teams[0],self.teams[1]]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    _roundRunning = False
            self.frame()
            if self.teams[0] == 0 and self.teams[1] == 0:
                # print("rematch")
                print("rematch")
                return Instance(self.models[self.model0dir][0], self.models[self.model1dir][0], tanksPerRound, self.models, self.model0dir, self.model1dir).evaluate()
            elif self.teams[0] <= tanksPerRound - killsForWin:
                # print("m1 wins")
                print("w")
                return [self.teams[0], self.teams[1]]
            elif self.teams[1] <= tanksPerRound - killsForWin:
                # print("m2 wins")
                print("w")
                return [self.teams[0], self.teams[1]]

    def updateModelsScore(self):
        _result = self.evaluate()
        if _result[0] > _result[1]:
            self.models[self.model0dir][1] += 1
            self.models[self.model1dir][1] -= 1
        elif _result[0] < _result[1]:
            self.models[self.model1dir][1] += 1
            self.models[self.model0dir][1] -= 1
        elif _result[0] == _result[1]:
            self.models[self.model1dir][1] -= 1
            self.models[self.model0dir][1] -= 1


def generation(models, ntanks):
    _models = copy.deepcopy(models)
    instances = []

    for i in range(len(_models)):
        for j in range(len(_models)-i-1):
            k = i + j + 1
            instances.append(Instance(_models[i][0],_models[k][0],ntanks, _models,i,k))

    threads = []

    for i in range(len(instances)):
        threads.append(threading.Thread(target=instances[i].updateModelsScore))

    for i in range(len(threads)):
        threads[i].start()
        #print("thead",i,"started")

    for i in range(len(threads)):
        threads[i].join()

    return _models


def cloneGeneration(models):
    _models = copy.deepcopy(models)
    _groups = []
    while(len(_models)>2):
        _groups.append([_models[0],_models[1],_models[2]])
        for i in range(3):
            _models.pop(0)

    for i in range(len(_groups)):
        _worstModelf = _groups[i][0][1]
        _worstModeli = 0
        for j in range(2):
            if _groups[i][j+1][1] < _worstModelf:
                _worstModelf = _groups[i][j+1][1]
                _worstModeli = j+1

        _groups[i].pop(_worstModeli)

        _p1 = _groups[i][0][0]
        _p2 = _groups[i][1][0]

        _newModel = Network([(nBodyLidar + nTurretLidar) * 6 + nRecurrence + 4] + networkShape + [7 + nRecurrence],Activation_ReLU)
        for i_ in range(_newModel.nlayers):
            for j in range(len(_newModel.layers[i_].weights)):
                for k in range(len(_newModel.layers[i_].weights[j])):
                    if np.random.rand() > 0.5:
                        _newModel.layers[i_].weights[j][k] = copy.deepcopy(_p1.layers[i_].weights[j][k])
                    else:
                        _newModel.layers[i_].weights[j][k] = copy.deepcopy(_p2.layers[i_].weights[j][k])
            for j in range(len(_newModel.layers[i_].biases)):
                for k in range(len(_newModel.layers[i_].biases[j])):
                    if np.random.rand() > 0.5:
                        _newModel.layers[i_].biases[j][k] = copy.deepcopy(_p1.layers[i_].biases[j][k])
                    else:
                        _newModel.layers[i_].biases[j][k] = copy.deepcopy(_p2.layers[i_].biases[j][k])

        _newModel.child(evolutionRate)

        _models.append([_p1,0])
        _models.append([_p2,0])
        _models.append([_newModel,0])

    _newModels = []
    while len(_models) != 0:
        _randint = np.random.randint(0,len(_models))
        _newModels.append(_models[_randint])
        _models.pop(_randint)

    return _newModels




if loadFrom == 0 or not os.path.exists(loadFrom):
    models = []
    for i in range(nModels):
        models.append([Network([(nBodyLidar + nTurretLidar) * 6 + nRecurrence + 4] + networkShape + [7 + nRecurrence],
                               Activation_ReLU),0])
else:
    models = pickle.load(open(loadFrom, "rb"))


if training == 1:
    _m2 = models

    for i in range(10000):
        print("generation=",i)
        _m1 = generation(_m2,tanksPerRound)
        #print(_m1)
        _m2 = cloneGeneration(_m1)
        #print(_m2)
        pickle.dump(_m2, open(saveTo, "wb"))

    pickle.dump(_m2, open(saveTo, "wb"))
    pickle.dump(_m2, open("backup1", "wb"))
else:

    models = pickle.load(open(loadFrom, "rb"))
    Instance(models[0][0],models[1][0],1,models,0,1).evaluate()

#model1 = Network([(nBodyLidar + nTurretLidar) * 6 + nRecurrence + 4] + networkShape + [7 + nRecurrence],
#                               Activation_ReLU)

#model2 = Network([(nBodyLidar + nTurretLidar) * 6 + nRecurrence + 4] + networkShape + [7 + nRecurrence],
#                               Activation_ReLU)

#Instance1 = Instance(model1,model2,4)
#Instance2 = Instance(model1,model2,4)
#T1 = threading.Thread(target=Instance1.evaluate)
#T2 = threading.Thread(target=Instance2.evaluate)

#T1.start()
#T2.start()


