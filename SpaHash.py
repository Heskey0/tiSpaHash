import taichi as ti
import taichi.math as tm

ti.init()

numSubsteps = 10
dt = 1.0 / 60.0 / numSubsteps


num_particles = 10000
radius = 0.02
radius_world = 5.0
radius_search = 2.0 * radius
pos_particles = ti.Vector.field(3, ti.f32, num_particles)
vel_particles = ti.Vector.field(3, ti.f32, num_particles)

querySize = ti.field(dtype=ti.i32, shape=())

@ti.data_oriented
class Hash:
    # spacing: 格子的大小
    def __init__(self, spacing, maxNumObjects):
        self.spacing = spacing
        self.tableSize = 2*maxNumObjects
        self.cellStart = ti.field(dtype=ti.i32, shape=(self.tableSize+1))
        self.cellEntries = ti.field(dtype=ti.i32, shape=(maxNumObjects))
        # 搜索到的物体集合
        self.queryIds = ti.field(dtype=ti.i32, shape=(maxNumObjects))

    @ti.func
    def hashCoords(self, xi, yi, zi):
        h = (int(xi) * 92837111) ^ (int(yi) * 689287499) ^ (int(zi) * 283923481)
        return ti.abs(h) % self.tableSize

    @ti.func
    def intCoord(self, coord):
        return ti.floor(coord / self.spacing)

    @ti.func
    def hashPos(self, pos, idx):
        return self.hashCoords(self.intCoord(pos[idx].x),
                               self.intCoord(pos[idx].y),
                               self.intCoord(pos[idx].z))

    @ti.kernel
    def create(self, pos: ti.template(), length: ti.template()):
        numObjects = length
        self.cellStart.fill(0)
        self.cellEntries.fill(0.0)

        for i in range(numObjects):
            h = self.hashPos(pos, i)
            self.cellStart[h] += 1

        # determine cells starts

        start = 0
        for i in range(self.tableSize):
            start += self.cellStart[i]
            self.cellStart[i] = start

        self.cellStart[self.tableSize] = start # guard

        # fill in objects ids

        for i in range(numObjects):
            h = self.hashPos(pos, i)
            self.cellStart[h] -= 1
            self.cellEntries[self.cellStart[h]] = i

    @ti.func
    def query(self, pos, idx, maxDist):
        x0 = self.intCoord(pos[idx].x - maxDist)
        y0 = self.intCoord(pos[idx].y - maxDist)
        z0 = self.intCoord(pos[idx].z - maxDist)

        x1 = self.intCoord(pos[idx].x + maxDist)
        y1 = self.intCoord(pos[idx].y + maxDist)
        z1 = self.intCoord(pos[idx].z + maxDist)


        for xi in range(x0, x1+1):
            for yi in range(y0, y1 + 1):
                for zi in range(z0, z1 + 1):
                    h = self.hashCoords(xi, yi, zi)
                    start = self.cellStart[h]
                    end = self.cellStart[h + 1]

                    for i in range(start, end+1):
                        self.queryIds[querySize[None]] = self.cellEntries[i]
                        querySize[None] = querySize[None] + 1


hash = Hash(radius_search, num_particles)
hash.create(pos_particles, num_particles)


@ti.kernel
def update_pos():
    for i in range(num_particles):
        pos_particles[i] = pos_particles[i] + vel_particles[i] * dt

# world collision
@ti.kernel
def world_col():
    for i in range(num_particles):
        if pos_particles[i].norm() > radius_world:
            dir_n = pos_particles[i] / pos_particles[i].norm()
            dir_v = vel_particles[i] / vel_particles[i].norm()
            dir_reflect = dir_v - 2.0 * dir_v.dot(dir_n) * dir_n
            pos_particles[i] = radius_world * dir_n
            vel_particles[i] = dir_reflect * vel_particles[i].norm()

@ti.kernel
def particle_col():
    for i in range(num_particles):
        querySize[None] = 0
        hash.query(pos_particles, i, radius_search)

        for _j in range(querySize[None]):
            j = hash.queryIds[_j]
            if i == j:
                break
            pos0 = pos_particles[i]
            vel0 = vel_particles[i]
            pos1 = pos_particles[j]
            vel1 = vel_particles[j]
            dis = (pos0 - pos1).norm()
            if dis < radius_search:
                dir = (pos0 - pos1) / dis

                # print(i, " :collid: ", j)
                # correct position
                corr = (radius_search-dis) * 0.5
                pos_particles[i] = pos0 + corr*dir
                pos_particles[j] = pos1 - corr*dir
                # correct velocity
                vi = vel0.dot(dir)
                vj = vel1.dot(dir)

                vel_particles[i] = (vj-vi)*(vel0 + dir)
                vel_particles[j] = (vi-vj)*(vel1 + dir)
                vel_particles[i] = min(vel_particles[i].norm(), vel0.norm())*vel_particles[i]/vel_particles[i].norm()
                vel_particles[j] = min(vel_particles[j].norm(), vel1.norm())*vel_particles[j]/vel_particles[j].norm()
                # print("vel i:", vel_particles[i], "vel j:", vel_particles[j])


def substep():
    update_pos()
    world_col()
    particle_col()



@ti.kernel
def init_pos_vel():
    for i in range(num_particles):
        pos_particles[i] = radius_world*0.8*ti.Vector([ti.random(), ti.random(), ti.random()])
        vel_particles[i] = 0.5*ti.Vector([ti.random(), ti.random(), ti.random()])


# ---------------------------------------------------------------------------- #
#                                      gui                                     #
# ---------------------------------------------------------------------------- #
# init the window, canvas, scene and camerea
window = ti.ui.Window("SpaHash", (1024, 1024), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
canvas.set_background_color((0.067, 0.184, 0.255))

# initial camera position
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)





def main():
    init_pos_vel()
    while window.running:
        # do the simulation in each step
        for _ in range(numSubsteps):
            substep()

        # set the camera, you can move around by pressing 'wasdeq'
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # set the light
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.ambient_light((0.5, 0.5, 0.5))

        # draw
        scene.particles(pos_particles, radius=radius, color=(0, 1, 1))

        # show the frame
        canvas.scene(scene)
        window.show()


if __name__ == '__main__':
    main()