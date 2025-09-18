import sys, json, time
import numpy as np
from stl import mesh as stl_mesh  # <- para ler STL
from PyQt5 import QtWidgets, QtCore
import pyqtgraph.opengl as gl

ANGLE_FILE = "data/angle.json"
MODEL_PATH = "roboanimacao.stl"  # caminho do STL

def ler_angulo(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if all(k in data for k in ('x','y','z')):
            return data
        else:
            return {'x':0, 'y':0, 'z':0}
    except:
        return {'x':0, 'y':0, 'z':0}

class STLApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Cria a janela OpenGL
        self.view = gl.GLViewWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

        # Adiciona um grid
        g = gl.GLGridItem()
        g.scale(2,2,1)
        self.view.addItem(g)

        # Carrega o STL
        self.stl_mesh = stl_mesh.Mesh.from_file(MODEL_PATH)
        verts = self.stl_mesh.vectors.reshape(-1,3)
        faces = np.arange(len(verts)).reshape(-1,3)
        colors = np.array([[0.5, 0.5, 1, 1] for _ in faces])  # azul claro

        self.mesh = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=colors, smooth=False, drawEdges=True)
        self.mesh.scale(0.1,0.1,0.1)  # ajuste de escala dependendo do STL
        self.view.addItem(self.mesh)

        # Timer para atualização
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_mesh)
        self.timer.start(20)  # ~50 FPS

        self.last_angle = {'x':0, 'y':0, 'z':0}

    def update_mesh(self):
        angle = ler_angulo(ANGLE_FILE)

        dx = angle['x'] - self.last_angle['x']
        dy = angle['y'] - self.last_angle['y']
        dz = angle['z'] - self.last_angle['z']

        self.mesh.rotate(dx, 1,0,0)
        self.mesh.rotate(dy, 0,1,0)
        self.mesh.rotate(dz, 0,0,1)

        self.last_angle = angle

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = STLApp()
    w.show()
    sys.exit(app.exec_())
