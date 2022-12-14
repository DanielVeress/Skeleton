//=============================================================================================
// Mintaprogram: Z?ld h?romsz?g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Veress Daniel
// Neptun : C8P32R
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const char * const vertexSource = R"(
	#version 330							
	precision highp float;					

	uniform mat4 MVP;						
	layout(location = 0) in vec2 vp;	

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		
	}
)";

const char * const fragmentSource = R"(
	#version 330			
	precision highp float;	
	
	uniform vec3 color;		
	out vec4 outColor;		

	void main() {
		outColor = vec4(color, 1);
	}
)";

struct Camera {
	float wCx, wCy;
	float wWx, wWy;
public:
	Camera() {
		Animate(0);
	}

	mat4 V() { 
		return mat4(1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0,
					-wCx, -wCy, 0, 1);
	}

	mat4 P() {
		return mat4(2 / wWx, 0, 0, 0,
					0, 2 / wWy, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1);
	}

	mat4 Vinv() {
		return mat4(1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0,
					wCx, wCy, 0, 1);
	}

	mat4 Pinv() {
		return mat4(wWx / 2, 0, 0, 0,
					0, wWy / 2, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1);
	}

	void Animate(float t) {
		wCx = 0;
		wCy = 0;
		wWx = 20;
		wWy = 20;
	}
};
#pragma endregion

Camera camera;
GPUProgram gpuProgram; 
const int nTesselatedVertices = 100;


class CatmullRom {
	float tension = -1;	
	std::vector<vec4> wCps;
	std::vector<float> ts;

	vec4 Hermite(vec4 p0, vec4 v0, float t0, vec4 p1, vec4 v1, float t1, float t) {
		vec4 a0 = p0;
		vec4 a1 = v0;
		vec4 a2 = 3 * (p1 - p0) / (powf((t1 - t0), 2)) - (v1 + 2 * v0) / (t1 - t0);
		vec4 a3 = 2 * (p0 - p1) / (powf((t1 - t0), 3)) + (v1 + v0) / (powf((t1 - t0), 2));

		vec4 result = a3 * powf(t - t0, 3) + a2 * powf(t - t0, 2) + a1 * (t - t0) + a0;
		return result;
	};
public:
	void AddControlPoint(vec4 wVertex) {
		float ti = wCps.size();

		wCps.push_back(wVertex);
		ts.push_back(ti);
	}

	vec4 r(float t) {
		for (int i = 0; i < wCps.size() + 1; i++) {
			int prev = i - 1, next = i + 1, nextnext = i + 2;

			float ts0, ts2, ts3;
			if (prev < 0) {
				prev = wCps.size() - 1;
				ts0 = -1;
			}
			else
				ts0 = ts[prev];

			if (next > wCps.size() - 1) {
				next = 0;
				ts2 = ts[ts.size() - 1] + 1;
				nextnext = 1;
				ts3 = ts2 + 1;
			}
			else {
				ts2 = ts[next];
				ts3 = ts[nextnext];
			}

			if (nextnext > wCps.size() - 1) {
				nextnext = 0;
				ts3 = ts[ts.size() - 1] + 1;
			}

			if (ts[i] <= t && t <= ts2) {

				vec4  p0 = wCps[prev], p1 = wCps[i], p2 = wCps[next], p3 = wCps[nextnext];
				float ts1 = ts[i];

				vec4 v1 = vec4(1, 1, 0, 0), v2 = vec4(1, 1, 0, 0);

				v1 = (1 - tension) / 2 * ((p2 - p1) / (ts2 - ts1) + (p1 - p0) / (ts1 - ts0));
				v2 = (1 - tension) / 2 * ((p3 - p2) / (ts3 - ts2) + (p2 - p1) / (ts2 - ts1));

				return Hermite(p1, v1, ts1, p2, v2, ts2, t);
			}
		}
	}

	void Clear() {
		wCps.clear();
		ts.clear();
	}
};

class SimplePolygon {
	unsigned int vaoPolygon, vboPolygon;
	unsigned int vaoCtrlPoints, vboCtrlPoints;
	std::vector<vec4> wCps;

	CatmullRom* curve = new CatmullRom();

public:
	SimplePolygon() {
		glGenVertexArrays(1, &vaoPolygon);
		glBindVertexArray(vaoPolygon);

		glGenBuffers(1, &vboPolygon);
		glBindBuffer(GL_ARRAY_BUFFER, vboPolygon);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL);

		glGenVertexArrays(1, &vaoCtrlPoints);
		glBindVertexArray(vaoCtrlPoints);
		glGenBuffers(1, &vboCtrlPoints);
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
	}

	void AddControlPoint(float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCps.push_back(wVertex);
	}

	int PickControlPoint(float cX, float cY) {
		vec4 mouse = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		for (unsigned int p = 0; p < wCps.size(); p++)
			if (dot(wCps[p] - mouse, wCps[p] - mouse) < 0.1) 
				return p;

		if (wCps.size() < 2)
			return -1;

		float minimalDistanceToLine = 99999;
		float minimalDistanceToPoint = 99999;
		int bestIdx = 0;

		for (int i = 0; i < wCps.size(); i++) {
			vec4 pointA, pointB;

			if (i == wCps.size() - 1) {
				pointA = wCps[i];
				pointB = wCps[0];
			} else {
				pointA = wCps[i];
				pointB = wCps[i+1];
			}

			vec4 lineVec = pointB - pointA;
			vec4 mouseVec = mouse - pointA;

			float shadowLength = (dot(mouseVec, lineVec) / dot(lineVec, lineVec));

			if (0 < shadowLength && shadowLength < 1) {
				vec4 closestPointToMouse = pointA + (lineVec * shadowLength);
				float distance = dot(closestPointToMouse - mouse, closestPointToMouse - mouse);

				if (minimalDistanceToLine > distance && minimalDistanceToPoint > distance) {
					minimalDistanceToLine = distance;
					if (i < wCps.size() - 1)
						bestIdx = i + 1;
					else
						bestIdx = 0;
				}
			} else {
				float pointDistance = dot(pointA - mouse, pointA - mouse);

				if (minimalDistanceToPoint > pointDistance && minimalDistanceToLine > pointDistance) {
					minimalDistanceToPoint = pointDistance;
					if (i < wCps.size() - 1)
						bestIdx = i;
					else
						bestIdx = 0;
				}
			}
		}

		wCps.insert(wCps.begin() + bestIdx, mouse);

		return bestIdx;
	}

	void MoveControlPoint(int p, float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCps[p] = wVertex;
	}

	void Refine() {
		for (int i = 0; i < wCps.size(); i++) {
			curve->AddControlPoint(wCps[i]);
		}

		std::vector<vec4> wCpsRef;
		for (float t = 0.5; t < wCps.size(); t++) {
			wCpsRef.push_back(wCps[t-0.5]);
			vec4 newPoint = curve->r(t);
			wCpsRef.push_back(newPoint);
		}
		curve->Clear();

		wCps = wCpsRef;
	}

	void Simplify() {
		int removeNum = floor(wCps.size() / 2.0);
		if (wCps.size() - removeNum <= 2)
			return;

		for (int rp = 0; rp < removeNum; rp++) {

			int bestIdx = 0;
			float minimalDistanceToLine = 99999;

			for (int i = 0; i < wCps.size(); i++) {
				vec4 pointPrev, pointCurrent, pointNext;

				if (i == 0) {
					pointPrev = wCps[wCps.size() - 1];
					pointCurrent = wCps[i];
					pointNext = wCps[i + 1];
				}
				else if (i == wCps.size() - 1) {
					pointPrev = wCps[i - 1];
					pointCurrent = wCps[i];
					pointNext = wCps[0];
				}
				else {
					pointPrev = wCps[i - 1];
					pointCurrent = wCps[i];
					pointNext = wCps[i + 1];
				}

				vec4 lineVec = pointPrev - pointNext;
				vec4 mouseVec = pointCurrent - pointNext;

				float shadowLength = (dot(mouseVec, lineVec) / dot(lineVec, lineVec));

				if (0 < shadowLength && shadowLength < 1) {
					vec4 closestPointToMouse = pointNext + (lineVec * shadowLength);
					float distance = dot(closestPointToMouse - pointCurrent, closestPointToMouse - pointCurrent);

					if (minimalDistanceToLine > distance) {
						minimalDistanceToLine = distance;
						bestIdx = i;
					}
				}
			}

			wCps.erase(wCps.begin() + bestIdx);

		}
	}

	void Draw() {
		mat4 VPTransform = camera.V() * camera.P();

		gpuProgram.setUniform(VPTransform, "MVP");

		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");

		if (wCps.size() > 0) {	
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, wCps.size() * 4 * sizeof(float), &wCps[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);
			glPointSize(10.0f);
			glDrawArrays(GL_POINTS, 0, wCps.size());
		}

		if (wCps.size() >= 2) {
			std::vector<float> vertexData;
			for (unsigned int i = 0; i < wCps.size(); i++) {
				vertexData.push_back(wCps[i].x);
				vertexData.push_back(wCps[i].y);
			}

			glBindVertexArray(vaoPolygon);
			glBindBuffer(GL_ARRAY_BUFFER, vboPolygon);
			glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(vec4), &vertexData[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 1, 1);
			glDrawArrays(GL_LINE_LOOP, 0, wCps.size());
		}
	}
};

SimplePolygon* poly;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f);

	poly = new SimplePolygon();

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	poly->Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') {
		poly->Simplify();
	}
	else if (key == 's') {
		poly->Refine();
	}

	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) { }

int pickedControlPoint = -1;
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight;

		poly->AddControlPoint(cX, cY);
		glutPostRedisplay();
	}

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight;

		pickedControlPoint = poly->PickControlPoint(cX, cY);
		
		glutPostRedisplay();
	}

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP) {
		pickedControlPoint = -1;
	}
}

void onMouseMotion(int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if (pickedControlPoint >= 0) poly->MoveControlPoint(pickedControlPoint, cX, cY);
}

void onIdle() {
	glutPostRedisplay();
}