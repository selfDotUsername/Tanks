import pygame
import numpy as np
import copy
import pickle
import time

numberofTanks = 13

keepStats = 1
loadFrom = 0
saveTo = "save13"

fps =6000
window_width = 1900
window_height = 1000

tanksImages = ['tank_chassis .png', 'tank_turret.png']
tanksSize = [50, 50]

tanksVelocity = 5
tanksRotation = 0.1
turretRotation = 0.1

bulletsVelocity = 7
bulletsSize = [25, 25]
bulletsImage = 'bullet_tail.png'

tanksShotTime = 100

nLidarBody = 4
nLidarTurret = 1
lidarSizeRate = 1

networkShape = [20,10]
nRecurrence = 5

initialRate = 1.5
rateRate = 0
pRateto0 = 0

framesToUpdatePos = 3000

pRandomTank = 0

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


class tank:
    tanks = []
    nTanks = 0
    totalTanks = 0

    def __init__(self, position, direction, turretDirection, lidarSize, nLidar, nTurretLidar, model, nrecurrence, rate):
        self.n = len(tank.tanks)
        tank.tanks.append(self)
        tank.nTanks = len(tank.tanks)
        tank.totalTanks += 1

        self.alive = 1

        self.score = 0

        self.X = position
        self.d = direction
        self.td = turretDirection
        self.v = tanksVelocity
        self.dv = tanksRotation
        self.tdv = turretRotation

        self.vsind = self.v * np.sin(self.d)
        self.vcosd = self.v * np.cos(self.d)
        self.sintd = np.sin(self.td)
        self.costd = np.cos(self.td)

        self.mUp = 0
        self.mDown = 0
        self.mLeft = 0
        self.mRight = 0
        self.mtLeft = 0
        self.mtRight = 0

        self.size = tanksSize
        self.body = pygame.transform.scale(pygame.image.load(tanksImages[0]), self.size)
        self.turret = pygame.transform.scale(pygame.image.load(tanksImages[1]), self.size)
        self.dDeg = self.d / (2 * np.pi) * 360
        self.tdDeg = self.td / (2 * np.pi) * 360

        self.lidarSize = lidarSize
        self.lidarSize2 = self.lidarSize ** 2
        self.nLidar = nLidar
        self.nTurretLidar = nTurretLidar

        self.model = model
        self.rate = rate
        self.recurrence = zeros(nrecurrence)
        self.nRecurrence = nrecurrence

        self.shotTime = tanksShotTime

        self.cloning = 0
        self.deletion = 0

    def draw(self):
        screen.blit(rot_center(self.body, -self.dDeg - 90),
                    (self.X[0] - self.size[0] / 2, self.X[1] - self.size[1] / 2))
        screen.blit(rot_center(self.turret, -self.tdDeg - 90),
                    (self.X[0] - self.size[0] / 2, self.X[1] - self.size[1] / 2))
        # pygame.draw.circle(screen, (255,0,0), (self.X[0] + 10*np.cos(self.td), self.X[1] + 10*np.sin(self.td)), 10)
        pygame.draw.circle(screen, (255, 0, 0), (self.X[0], self.X[1]), 2)

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
        self.dDeg = self.d / (2 * np.pi) * 360

    def right(self):
        self.d += self.dv
        self.d = self.d % (2 * np.pi)
        self.vsind = self.v * np.sin(self.d)
        self.vcosd = self.v * np.cos(self.d)
        self.dDeg = self.d / (2 * np.pi) * 360

    def tLeft(self):
        self.td -= self.tdv
        self.td = self.td % (2 * np.pi)
        self.sintd = np.sin(self.d)
        self.costd = np.cos(self.d)
        self.tdDeg = self.td / (2 * np.pi) * 360

    def tRight(self):
        self.td += self.tdv
        self.td = self.td % (2 * np.pi)
        self.sintd = np.sin(self.d)
        self.costd = np.cos(self.d)
        self.tdDeg = self.td / (2 * np.pi) * 360

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
        if self.X[0] > window_width - self.size[0]/2:
            self.X[0] = window_width - self.size[0]/2
        if self.X[0] < self.size[0]/2:
            self.X[0] = self.size[0]/2
        if self.X[1] > window_height - self.size[1]/2:
            self.X[1] = window_height - self.size[1]/2
        if self.X[1] < self.size[1]/2:
            self.X[1] = self.size[1]/2


    def shoot(self):
        if self.shotTime < 1:
            bullet.totalBullets += 1
            bullet(self)
            self.shotTime = tanksShotTime


    def lidarWalls(self,d):
        _x, _y = self.X[0], self.X[1]
        if d == 0:
            # right
            # print(1)
            return [(window_width - _x) ** 2, 0, 0]
        if d == np.pi / 2:
            # bottom
            # print(2)
            return [_y ** 2, 0, 0]
        if d == np.pi:
            # left
            # print(3)
            return [_x ** 2, 0, 0]
        if d == 3 * np.pi / 2:
            # top
            # print(4)
            return [(window_height - _y) ** 2, 0, 0]

        _m = np.tan(d)

        if d < np.pi / 2:
            if _m * (window_width - _x) + _y < window_height:
                # right
                # print(5)
                return [squaredistance(_x, _y, window_width, _m * (window_width - _x) + _y), 0, 0]
            else:
                # bottom
                # print(6)
                return [squaredistance(_x, _y, _x + (window_height - _y) / _m, window_height), 0, 0]
        if d < np.pi:
            if _y - _m * _x <= window_height:
                # left
                # print(7)
                return [squaredistance(_x, _y, 0, _y - _m * _x), 0, 0]
            else:
                # bottom
                # print(8)
                return [squaredistance(_x, _y, _x + (window_height - _y) / _m, window_height), 0, 0]
        if d < 3 * np.pi / 2:
            if _y - _m * _x >= 0:
                # left
                # print(9)
                return [squaredistance(_x, _y, 0, _y - _m * _x), 0, 0]
            else:
                # top
                # print(10)
                return [squaredistance(_x, _y, _x - _y / _m, 0), 0, 0]
        if _x - _y / _m > window_width:
            # right
            # print(11)
            return [squaredistance(_x, _y, window_width, _m * (window_width - _x) + _y), 0, 0]
        else:
            # top
            # print(12)
            return [squaredistance(_x, _y, _x - _y / _m, 0), 0, 0]


    def lidarSingle(self, d):
        #print("d=",d)
        _lidarOutput = []
        _tDetections = []
        _sin = np.sin(d)
        _cos = np.cos(d)
        for i in range(len(tank.tanks)):
            if tank.tanks[i] is self:
                _tDetections.append(None)
            else:
                _tDetections.append(isLidarCollision(self.X, tank.tanks[i].X, self.lidarSize2, _sin, _cos))
        _bDetections = []
        for i in range(len(bullet.bullets)):
            _bDetections.append(isLidarCollision(self.X, bullet.bullets[i].X, self.lidarSize2, _sin, _cos))
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
                return _lidarWalls + _lidarWalls
            if _bmin is None:
                return [_tmin, tank.tanks[_tmini].td, tank.tanks[_tmini].d] + _lidarWalls
            if _tmin is None:
                return _lidarWalls + [_bmin, bullet.bullets[_bmini].d, 0]
        return [_tmin, tank.tanks[_tmini].td, tank.tanks[_tmini].d] + [_bmin, bullet.bullets[_bmini].d, 0]


    def lidarAll(self):
        _lidarList = []
        _directionStep = (2 * np.pi) / (self.nLidar)
        for i in range(self.nLidar):
            _singleLidar = self.lidarSingle((self.d + i * _directionStep) % (2 * np.pi))
            _lidarList += _singleLidar
        _directionStep = (2 * np.pi) / (self.nTurretLidar)
        for i in range(self.nTurretLidar):
            _singleLidar = self.lidarSingle((self.td + i * _directionStep)%(2*np.pi))
            _lidarList += _singleLidar
        return _lidarList

    def collectAiData(self):
        return self.lidarAll() + [self.d, + self.td, self.shotTime] + self.recurrence

    def forwardModel(self):
        _networkInput = self.collectAiData()
        self.model.forward([_networkInput])
        _modelOutput = self.model.output[0]

        if _modelOutput[0]> 0:
            self.mUp = 1
        else:
            self.mUp = 0

        if _modelOutput[1]> 0:
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

        for i in range(self.nRecurrence):
            self.recurrence[i] = (_modelOutput[7+i]%100)

    def clone(self):
        #if len(tank.tanks) >= numberofTanks:
        #    return None
        _newModel = copy.deepcopy(self.model)
        _newModel.child(self.rate)
        _randomx = np.random.rand() * window_width
        _randomy = np.random.rand() * window_height
        _randomd = np.random.rand() * 2 * np.pi
        if np.random.rand() < pRateto0:
            _newRate = 0
        else:
            _newRate = self.rate + random(-0.1, 0.1) * rateRate

        _newLidarSize = self.lidarSize + random(-1,1) * lidarSizeRate
        print(_newLidarSize)
        tank([_randomx,_randomy],_randomd,_randomd,_newLidarSize,self.nLidar,self.nTurretLidar,_newModel,self.nRecurrence,_newRate)

    def brainCombo(self,p1,p2):
        for i in range(self.model.nlayers):
            for j in range(len(self.model.layers[i].weights)):
                for k in range(len(self.model.layers[i].weights[j])):
                    if np.random.rand() > 0.5:
                        self.model.layers[i].weights[j][k] = copy.deepcopy(p1.model.layers[i].weights[j][k])
                    else:
                        self.model.layers[i].weights[j][k] = copy.deepcopy(p2.model.layers[i].weights[j][k])
            for j in range(len(self.model.layers[i].biases)):
                for k in range(len(self.model.layers[i].biases[j])):
                    if np.random.rand() > 0.5:
                        self.model.layers[i].biases[j][k] = copy.deepcopy(p1.model.layers[i].biases[j][k])
                    else:
                        self.model.layers[i].biases[j][k] = copy.deepcopy(p2.model.layers[i].biases[j][k])

    def clone2(self, otherTank):
        if random(0,1) < pRandomTank:
            _initialModel = Network(
                [(nLidarBody + nLidarTurret) * 6 + nRecurrence + 3] + networkShape + [7 + nRecurrence],
                Activation_ReLU)
            _randomX = random(0, 1) * window_width
            _randomY = random(0, 1) * window_height
            _randomD = random(0,2*np.pi)
            tank([_randomX, _randomY], _randomD, 10, 10, nLidarTurret, nLidarBody, _initialModel, nRecurrence, initialRate)
            return 0

        if len(tank.tanks) < numberofTanks+2:
            _newModel = copy.deepcopy(self.model)
            _newModel.child(self.rate)
            #creature.totalCreatures += 1
        # print(self.rate)
            _randomx = np.random.rand() * window_width
            _randomy = np.random.rand() * window_height
            _randomd = np.random.randn() * 2 * np.pi
            #print(self.rate, self.lidarSize)
            #_newRate = self.rate + random(-0.1, 0.1) * creaturesRateRate
            #if np.random.rand() > pRateTo0:
            #    _newRate = self.score/10
            #print(self.lidarSize)
            _newTank = tank([_randomx,_randomy],_randomd,_randomd,10, self.nLidar, self.nTurretLidar, self.model, self.nRecurrence, initialRate)
            #_newCreature = creature([_randomx, _randomy], _randomd, 10, 0.1, 10, (100, 100, 100), creaturesnLidar,
            #     self.lidarSize + random(-1, 1) * lidarSizeRate, creaturesStartingEnergy,
            #     _newModel, creaturesRecurrence, _newRate)
            _newTank.brainCombo(self,otherTank)
            _newTank.model.child(_newTank.rate)

    def doCloning(self):
        while self.cloning > 0:
            print(mostKillsTank().score)
            self.clone2(mostKillsTank())
            self.cloning -= 1

    def delete(self):
        if self.deletion == 1:
            for i in range(self.n + 1, len(tank.tanks)):
                tank.tanks[i].n -= 1
            tank.tanks.pop(self.n)

    def frame(self):
        if self.shotTime != 0:
            self.shotTime -= 1
        self.forwardModel()
        self.move()
        self.draw()


class bullet:
    shotsOnTarget = 0
    shotsMissed = 0
    bullets = []
    nBullets = 0
    totalBullets = 0

    def __init__(self, tank0):
        self.n = len(bullet.bullets)
        bullet.bullets.append(self)
        bullet.nBullets = len(bullet.bullets)

        self.tank = tank0

        self.X = copy.deepcopy(tank0.X)
        self.v = bulletsVelocity
        self.image = pygame.transform.scale(pygame.image.load(bulletsImage), bulletsSize)
        self.size = bulletsSize
        self.d = copy.deepcopy(tank0.td)
        self.dDeg = self.d / (2 * np.pi) * 360

        self.vsind = self.v * np.sin(self.d)
        self.vcosd = self.v * np.cos(self.d)

        self.deletion = 0

    def draw(self):
        screen.blit(rot_center(self.image, -self.dDeg - 90),
                    (self.X[0] - self.size[0] / 2, self.X[1] - self.size[1] / 2))

    def move(self):
        self.X[0] += self.vcosd
        self.X[1] += self.vsind
        pass

    def delete(self):
        if self.deletion == 1:
            for i in range(self.n + 1, len(bullet.bullets)):
                bullet.bullets[i].n -= 1
            bullet.bullets.pop(self.n)
            if self.tank not in tank.tanks:
                self.tank.doCloning()

    def collisionDetection(self):
        for i in range(len(tank.tanks)):
            if tank.tanks[i] is not self.tank and isL1Collisoin(self.X, tank.tanks[i].X,
                                                                (self.size[0] + tank.tanks[i].size[0]) / 2) and tank.tanks[i].alive == 1:
                self.deletion = 1
                tank.tanks[i].deletion = 1
                tank.tanks[i].alive = 0
                self.tank.cloning += 1
                bullet.shotsOnTarget += 1
                self.tank.score += 1
        if self.X[0] > window_width or self.X[0] < 0 or self.X[1] > window_height or self.X[1] < 0:
            self.deletion = 1
            bullet.shotsMissed += 1

    def frame(self):
        self.move()
        self.collisionDetection()
        self.draw()


def updateBullets():
    for i in range(len(bullet.bullets)):
        bullet.bullets[i].frame()
    _deletions = 0
    for i in range(len(bullet.bullets)):
        _p = 0
        if bullet.bullets[i - _deletions].deletion == 1:
            _p = 1
            bullet.bullets[i - _deletions].delete()
        _deletions += _p


def updateTanks():
    for i in range(len(tank.tanks)):
        tank.tanks[i].frame()
    _deletions = 0
    for i in range(len(tank.tanks)):
        tank.tanks[i-_deletions].doCloning()
        if tank.tanks[i - _deletions].deletion == 1:
            tank.tanks[i - _deletions].delete()
            _deletions += 1


def playerControlls():
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_w:
            player.mUp = 1
        if event.key == pygame.K_a:
            player.mLeft = 1
        if event.key == pygame.K_s:
            player.mDown = 1
        if event.key == pygame.K_d:
            player.mRight = 1
        if event.key == pygame.K_LEFT:
            player.mtLeft = 1
        if event.key == pygame.K_RIGHT:
            player.mtRight = 1
        if event.key == pygame.K_UP:
            player.shoot()

    if event.type == pygame.KEYUP:
        if event.key == pygame.K_w:
            player.mUp = 0
        if event.key == pygame.K_a:
            player.mLeft = 0
        if event.key == pygame.K_s:
            player.mDown = 0
        if event.key == pygame.K_d:
            player.mRight = 0
        if event.key == pygame.K_LEFT:
            player.mtLeft = 0
        if event.key == pygame.K_RIGHT:
            player.mtRight = 0

def saveModels(directory):
    models = []
    for i in range(len(tank.tanks)):
        models.append([tank.tanks[i].model, tank.tanks[i].rate, tank.tanks[i].lidarSize])
    data = [models,tank.totalTanks]
    pickle.dump(data, open(directory, "wb"))

def loadModels(directory):
    data = pickle.load(open(directory, "rb"))
    if keepStats == 1:
        tank.totalTanks = data[1]
    for i in range(len(data[0])):
        _randomX = random(0,window_width)
        _randomY = random(0,window_height)
        tank([_randomX,_randomY], 0, 0, data[0][i][2], nLidarBody, nLidarTurret, data[0][i][0], nRecurrence, data[0][i][1])

def mostKillsTank():
    _c = tank.tanks[0]
    for i in range(len(tank.tanks)-1):
        if tank.tanks[i+1].score > _c.score:
            _c = tank.tanks[i+1]
    return  _c

def initialiseTanks(n):
    for i in range(n):
        _initialModel = Network([(nLidarBody + nLidarTurret) * 6 + nRecurrence + 3] + networkShape + [7 + nRecurrence],
                                Activation_ReLU)
        _randomX = random(0,1) * window_width
        _randomY = random(0,1) * window_height
        tank([_randomX, _randomY], 10, 10, 10, nLidarTurret, nLidarBody, _initialModel, nRecurrence, initialRate)


if loadFrom == 0:
    _initialModel = Network([(nLidarBody+nLidarTurret)*6+nRecurrence+3] + networkShape + [7+nRecurrence], Activation_ReLU) #
    player = tank([100, 100], 0, 0, 100, nLidarTurret, nLidarBody, _initialModel, nRecurrence, initialRate)
    initialiseTanks(numberofTanks)
    #print("b",(nLidarBody+nLidarTurret)*6+nRecurrence+2)
    #print("a",player.collectAiData())
else:
    loadModels(loadFrom)


_updatePosTimer = framesToUpdatePos
_summonedTanks = 0

running = True
while running:
    _updatePosTimer -= 1
    if _updatePosTimer < 1:
        for i in range(len(tank.tanks)):
            tank.tanks[i].X = [random(0,window_width),random(0,window_height)]
        _updatePosTimer = framesToUpdatePos
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        playerControlls()
        if event.type == pygame.QUIT:
            saveModels(saveTo)
            running = False
    #print(isLidarCollision(tank.tanks[0].X,tank.tanks[1].X,500,np.sin(tank.tanks[0].d),np.cos(tank.tanks[0].d)))
    #print(tank.tanks[0].lidarAll())
    # tank.tanks[0].tLeft()
    #print(tank.tanks[0].lidarSingle(tank.tanks[0].d))
    updateTanks()
    updateBullets()
    #if len(tank.tanks) < numberofTanks:
    #    tank.tanks[0].clone()
    #    _summonedTanks += 1
    screen.blit(font.render("Tanks:" + str(len(tank.tanks)), 1, pygame.Color("coral")), (10, 20))
    screen.blit(font.render("total tanks:" + str(tank.totalTanks), 1, pygame.Color("coral")), (10, 40))
    screen.blit(font.render("Shots on target:" + str(bullet.shotsOnTarget), 1, pygame.Color("coral")), (10, 60))
    screen.blit(font.render("Shots missed:" + str(bullet.shotsMissed), 1, pygame.Color("coral")), (10, 80))
    screen.blit(font.render("total bullets:" + str(bullet.totalBullets), 1, pygame.Color("coral")), (10, 100))

    #screen.blit(font.render("summoned tanks:" + str(_summonedTanks), 1, pygame.Color("coral")), (10, 60))
    screen.blit(update_fps(), (10, 0))
    pygame.display.update()
    clock.tick(fps)
    #print(tank.tanks[0].model.output[0])

