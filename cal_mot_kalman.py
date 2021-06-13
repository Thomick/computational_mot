from pdb import set_trace
import numpy as np
from numpy.lib.function_base import append
from make_scenes import Canvas,Scene,Objs,visualize_frames
#from filterpy.common import Q_discrete_white_noise

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MotKalman():
  def __init__(self,class_scene,num_targets=1,dt=1.,std_measure=0.3,std_pred=0.3,covP=100.):
    self.class_scene = class_scene
    self.num_measures = class_scene.max_frames
    self.num_targets = num_targets
    self.std_measure = std_measure
    self.std_pred = std_pred
    self.covP = covP
    self.dt =dt
    self.gt_pos = [] #ground truth positions
    self.K = []
    self.x = []
    self.P = [] 
    self.est_pos = np.zeros((self.num_targets,self.num_measures,4))

    self.u = np.array([[0.], [0.], [0.], [0.]]) # estimation noise
    self.F = np.array([[1., 0., self.dt, 0.], [0., 1., 0., self.dt], [0., 0., 1., 0.], [0., 0., 0., 1.]])  # state transition matrix
    self.H = np.array([[1., 0., 0, 0], [0., 1., 0., 0.]])  # to extract position coordinates
    self.R = np.array([[self.std_measure**2, 0], [0, self.std_measure**2]]) # measurement noise
    self.I = np.identity((len(self.u)))    # identity matrix

    self.extract_gt_pos() 
    self.init_kalmanparams_all()
    self.exec_kalmanfilter_all()

    #for visualize
    self.col_obj = [255,0,0]
    self.est_frames = np.ones((class_scene.output_frames.shape))*class_scene.output_frames

  def extract_gt_pos(self):
    self.gt_pos = [self.class_scene.output_centerpt[i] for i in range(self.num_measures)]

  def init_kalmanparams_all(self):
    [self.init_kalmanparams_eachobj(i) for i in range(self.num_targets)]

  def init_kalmanparams_eachobj(self,ind_obj):
    self.x.append(np.array([[self.gt_pos[0][ind_obj][0]], [self.gt_pos[0][ind_obj][1]], [0.], [0.]])) # initial [position x, position y,velocity x, velocity y]
    self.P.append(np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., self.covP, 0.], [0., 0., 0., self.covP]])) #covariance matrix

  def exec_kalmanfilter_all(self):
    [self.exec_kalmanfilter_eachobj(i) for i in range(self.num_targets)]
    
  def exec_kalmanfilter_eachobj(self,ind_obj):
    K_obj=[]
    x = np.array(self.x[ind_obj])
    P = np.array(self.P[ind_obj])
    for n in range(self.num_measures):       
      # prediction
      #q = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.std_pred)
      #self.u[2] = q[0,0]
      #self.u[3] = q[1,1]
 
      x = np.dot(self.F, x) + self.u
      P = np.dot(np.dot(self.F, P), self.F.T)

      # measurement
      z = np.array([self.gt_pos[n][ind_obj]])
      y = z.T - np.dot(self.H, x)
      S = np.dot(np.dot(self.H, P), self.H.T) + self.R
      K = np.dot(np.dot(P, self.H.T), np.linalg.inv(S))
      x = x + np.dot(K, y)        
      P = np.dot((self.I - np.dot(K, self.H)), P)
      K_obj.append(K)
      self.est_pos[ind_obj,n,:] = np.reshape(x,[x.shape[0]])

    self.x[ind_obj] = x.tolist()
    self.P[ind_obj] = P.tolist()
    self.K.append(K_obj)

  def make_estframes(self):
    for t in range(self.num_measures-1):
      [self.draw_circle_estimated(t+1,i) for i in range(self.num_targets)] 


  def draw_circle_estimated(self,ind_frame,ind_obj):
    w1=np.arange(0,self.class_scene.class_canvas.canvas_x,1)
    h1=np.arange(0,self.class_scene.class_canvas.canvas_y,1)
    wx,wy = np.meshgrid(w1,h1)
    wx = wx-(self.class_scene.class_canvas.center_x)
    wy = wy-(self.class_scene.class_canvas.center_y)
    wx = wx - (self.est_pos[ind_obj][ind_frame][0] - self.class_scene.class_canvas.center_x)
    wy = wy - (self.est_pos[ind_obj][ind_frame][1] - self.class_scene.class_canvas.center_y)
    r = np.sqrt(wx**2 + wy**2)

    tmp_r = self.est_frames[ind_frame,:,:,0]
    tmp_r[np.where(r<self.class_scene.stored_objs[ind_obj].diameter/2)] = self.col_obj[0]
    tmp_g = self.est_frames[ind_frame,:,:,1]
    tmp_g[np.where(r<self.class_scene.stored_objs[ind_obj].diameter/2)] = self.col_obj[1]
    tmp_b = self.est_frames[ind_frame,:,:,2]
    tmp_b[np.where(r<self.class_scene.stored_objs[ind_obj].diameter/2)] = self.col_obj[2]


def visualize_frames_est(class_motkalman,canvas_x,canvas_y,flag_save=False,fname_save='sample_est.gif'):
  fig= plt.figure()
  ax = fig.add_subplot(111)
  tmp = class_motkalman.est_frames.astype('uint8')
  imgs = [[ax.imshow(np.reshape(tmp[j,:,:,:],[canvas_y,canvas_x,3]))] for j in range(class_motkalman.class_scene.max_frames)]
  ani = animation.ArtistAnimation(fig,imgs,interval=1)
  if (flag_save==True):
    ani.save(fname_save, writer="pillow",fps=60)
  plt.show()

if __name__=='__main__':
  #Scene parameters
  num_obj = 3
  canvas_x = 256
  canvas_y = 256
  max_frames = 500
  col_obj = [255,255,255]
  diameter = 20
  speed = 2 #in pixesls per time
  direction_min= 0 #in degrees
  direction_max= 360 # in degrees
  ##############################
  #make scenes
  class_canvas = Canvas(canvas_x,canvas_y)
  class_baseobj = Objs(class_canvas,diameter,speed,direction_min,direction_max)
  class_scene = Scene(class_canvas,class_baseobj,max_frames,num_obj,col_obj)
  class_scene.update_scenes_all()

  #Kalman parameters
  dt = 1
  std_pred = 0.3
  std_measure = 0.3
  num_targets = 2
  covP = 200

  #Kalman parameters
  class_motkalman = MotKalman(class_scene,num_targets=num_targets,dt=dt,std_measure=std_measure,std_pred=std_pred,covP=covP)
  #def __init__(self,class_scene,num_targets=1,dt=1.,std_measure=0.3,std_pred=0.3,covP=100.):
  class_motkalman.make_estframes()


  flag_save = False
  fname_save_est = 'sample_mot_est.gif'
  visualize_frames_est(class_motkalman,canvas_x,canvas_y,flag_save,fname_save_est)