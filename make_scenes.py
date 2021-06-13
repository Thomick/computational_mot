import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Canvas():
  def __init__(self,canvas_x=256,canvas_y=256):
    self.canvas_x = canvas_x
    self.canvas_y = canvas_y
    self.center_x = int(canvas_x/2)
    self.center_y = int(canvas_y/2)


class Scene():
  def __init__(self,class_canvas,class_baseobj,max_frames=100,num_obj=1,col_obj = [255,255,255]):
    self.class_canvas = class_canvas
    
    #about time
    self.max_frames = max_frames
    self.current_frame = 0
    #about objects
    self.num_obj =num_obj
    self.stored_objs = []
    self.init_stored_centerpt = []
    self.tmp_centerpt = []
    self.output_centerpt = []
    self.col_obj = col_obj
    #initialize
    self.initialize_scenes(class_baseobj)


  def initialize_scenes(self,class_baseobj):
    #initialize output frames
    self.output_frames = np.zeros((self.max_frames,self.class_canvas.canvas_y,self.class_canvas.canvas_x,3))
    
    #make objects
    [self.add_newobj(class_baseobj) for i in range(self.num_obj)]
    #
    [self.draw_circle(0,i) for i in range(self.num_obj)]
    self.output_centerpt.append(self.init_stored_centerpt)

  def update_scene(self):
    self.current_frame += 1
    self.tmp_centerpt = []
    [self.stored_objs[i].update_pos() for i in range(self.num_obj)]
    [self.draw_circle(self.current_frame,i) for i in range(self.num_obj)]
    self.output_centerpt.append(self.tmp_centerpt)

  def update_scenes_all(self):
    for t in range(self.max_frames-self.current_frame-1):
      self.current_frame += 1
      self.tmp_centerpt = []
      [self.stored_objs[i].update_pos() for i in range(self.num_obj)]
      [self.draw_circle(self.current_frame,i) for i in range(self.num_obj)] 
      self.output_centerpt.append(self.tmp_centerpt)

  def add_newobj(self,class_baseobj):
    new_obj = Objs(self.class_canvas,
        class_baseobj.diameter,class_baseobj.speed,class_baseobj.direction_min,class_baseobj.direction_max)
    flag_loop = True
    flag_overlap = False
    while flag_loop==True:
      for j in range(len(self.stored_objs)):
        flag_overlap = False
        d = math.dist([new_obj.x,new_obj.y],[self.stored_objs[j].x,self.stored_objs[j].y])
        if (d < self.stored_objs[j].diameter):
          flag_overlap = True
          break
      if flag_overlap==False:
        self.stored_objs.append(new_obj)
        self.init_stored_centerpt.append([new_obj.x,new_obj.y]) 
        flag_loop = False

  def draw_circle(self,ind_frame,ind_obj):
    w1=np.arange(0,self.class_canvas.canvas_x,1)
    h1=np.arange(0,self.class_canvas.canvas_y,1)
    wx,wy = np.meshgrid(w1,h1)
    wx = wx-(self.class_canvas.center_x)
    wy = wy-(self.class_canvas.center_y)
    wx = wx - (self.stored_objs[ind_obj].x - self.class_canvas.center_x)
    wy = wy - (self.stored_objs[ind_obj].y - self.class_canvas.center_y)
    r = np.sqrt(wx**2 + wy**2)

    tmp_r = self.output_frames[ind_frame,:,:,0]
    tmp_r[np.where(r<self.stored_objs[ind_obj].diameter/2)] = self.col_obj[0]
    tmp_g = self.output_frames[ind_frame,:,:,1]
    tmp_g[np.where(r<self.stored_objs[ind_obj].diameter/2)] = self.col_obj[1]
    tmp_b = self.output_frames[ind_frame,:,:,2]
    tmp_b[np.where(r<self.stored_objs[ind_obj].diameter/2)] = self.col_obj[2]

    #center position stored
    self.tmp_centerpt.append([self.stored_objs[ind_obj].x,self.stored_objs[ind_obj].y])


class Objs():
  def __init__(self,class_canvas,diameter=50,speed=1,direction_min=10,direction_max=340):
    self.class_canvas = class_canvas
    self.diameter = diameter
    self.x = np.random.randint(int(self.diameter/2),self.class_canvas.canvas_x-int(self.diameter/2))
    self.y = np.random.randint(int(self.diameter/2),self.class_canvas.canvas_y-int(self.diameter/2))
    self.sign_x0 = 1  #plus 1 or minus 1
    self.sign_y0 = 1  #plus 1 or minus 1
    self.speed = speed 
    self.direction_min = direction_min
    self.direction_max = direction_max
    self.direction = direction_min + direction_max*np.random.rand() 
    self.vx = (self.sign_x0 * self.speed) * np.cos(np.deg2rad(self.direction))
    self.vy = (self.sign_y0 * self.speed) * np.sin(np.deg2rad(self.direction))

  def update_pos(self):
    self.x += (self.sign_x0 * self.speed) * np.cos(np.deg2rad(self.direction))
    self.y += (self.sign_y0 * self.speed) * np.sin(np.deg2rad(self.direction))
  
    if (np.abs(self.x-self.class_canvas.center_x) > (self.class_canvas.center_x-(self.diameter/2))):
      self.sign_x0 = -self.sign_x0
    if (np.abs(self.y-self.class_canvas.center_y) > (self.class_canvas.center_y-(self.diameter/2))):
      self.sign_y0 = -self.sign_y0


def visualize_frames(class_scene,canvas_x,canvas_y,flag_save=False,fname_save='sample.gif'):
  fig= plt.figure()
  ax = fig.add_subplot(111)
  tmp = class_scene.output_frames.astype('uint8')
  imgs = [[ax.imshow(np.reshape(tmp[j,:,:,:],[canvas_y,canvas_x,3]))] for j in range(class_scene.max_frames)]
  ani = animation.ArtistAnimation(fig,imgs,interval=1)
  #ani.save("sample.gif", writer="pillow")
  if (flag_save==True):
    ani.save(fname_save, writer="pillow",fps=60)
  plt.show()



if __name__=='__main__':
  num_obj = 5
  canvas_x = 256
  canvas_y = 256
  max_frames = 500
  col_obj = [255,255,255]
  diameter = 20
  speed = 4 #in pixesls per time
  direction_min=0 #in degrees
  direction_max=360 # in degrees

  class_canvas = Canvas(canvas_x,canvas_y)
  class_baseobj = Objs(class_canvas,diameter,speed,direction_min,direction_max)
  class_scene = Scene(class_canvas,class_baseobj,max_frames,num_obj,col_obj)
  class_scene.update_scenes_all()

  flag_save = False
  fname_save = 'sample_mot.gif'
  visualize_frames(class_scene,canvas_x,canvas_y,flag_save,fname_save)