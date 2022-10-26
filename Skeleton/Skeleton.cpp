#pragma region base
//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
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

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330							// Shader 3.3
	precision highp float;					// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;						// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;		// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
public:
	Camera() {
		Animate(0);
	}

	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Vinv() { // inverse view matrix
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}

	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void Animate(float t) {
		wCx = 0; // 10 * cosf(t);
		wCy = 0;
		wWx = 20;
		wWy = 20;
	}
};
#pragma endregion

Camera camera;	// 2D camera
GPUProgram gpuProgram; // vertex and fragment shaders
const int nTesselatedVertices = 100;

#pragma region polygon

class SimplePolygon {
	unsigned int vaoPolygon, vboPolygon;
	unsigned int vaoCtrlPoints, vboCtrlPoints;
	std::vector<vec4> wCps;

public:
	SimplePolygon() {
		// Curve
		glGenVertexArrays(1, &vaoPolygon);
		glBindVertexArray(vaoPolygon);

		glGenBuffers(1, &vboPolygon); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vboPolygon);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL); // attribute array, components/attribute, component type, normalize?, stride, offset

		// Control Points
		glGenVertexArrays(1, &vaoCtrlPoints);
		glBindVertexArray(vaoCtrlPoints);

		glGenBuffers(1, &vboCtrlPoints); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL); // attribute array, components/attribute, component type, normalize?, stride, offset
	}

	// Add a given 2D control point
	virtual void AddControlPoint(float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCps.push_back(wVertex);
	}

	// Returns the selected control point or -1
	int PickControlPoint(float cX, float cY) {
		vec4 mouse = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		for (unsigned int p = 0; p < wCps.size(); p++) {
			if (dot(wCps[p] - mouse, wCps[p] - mouse) < 0.1) 
				return p;
		}


		if (wCps.size() < 2) {
			return -1;
		}

		///																			TODO - - - Reformat for readability
		float minimalDistanceToLine = 99999;
		float minimalDistanceToPoint = 99999;
		int bestIdx = 0;
		// for every line we calculate the distance
		for (int i = 0; i < wCps.size(); i++) {
			vec4 pointA, pointB;

			if (i == wCps.size() - 1) {
				pointA = wCps[i];
				pointB = wCps[0];
			} else {
				pointA = wCps[i];
				pointB = wCps[i+1];
			}

			vec4 lineVec = pointB - pointA; // we get the vector that points from i to i+1
			vec4 mouseVec = mouse - pointA; // we get vector that points from i to mouse point

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

	// An indexed point is changed with new values
	void MoveControlPoint(int p, float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCps[p] = wVertex;
	}

	// Make the polygon more curvy
	void Refine() {
		
		

	}

	// deletes ~half of the points
	void Simplify() {
		int removeNum = floor(wCps.size() / 2.0);
		if (wCps.size() - removeNum <= 2)
			return;

		for (int rp = 0; rp < removeNum; rp++) {
			// for every point we calculate the distance
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

				vec4 lineVec = pointPrev - pointNext; // we get the vector that points from i to i+1
				vec4 mouseVec = pointCurrent - pointNext; // we get vector that points from i to mouse point

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

	// Draw everything
	void Draw() {
		mat4 VPTransform = camera.V() * camera.P();

		gpuProgram.setUniform(VPTransform, "MVP");

		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");

		// draw ctrl points
		if (wCps.size() > 0) {	
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, wCps.size() * 4 * sizeof(float), &wCps[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);
			glPointSize(10.0f);
			glDrawArrays(GL_POINTS, 0, wCps.size());
		}

		// draw plygon
		if (wCps.size() >= 2) {
			std::vector<float> vertexData;
			for (unsigned int i = 0; i < wCps.size(); i++) {
				vertexData.push_back(wCps[i].x);
				vertexData.push_back(wCps[i].y);
			}

			// copy data to the GPU
			glBindVertexArray(vaoPolygon);
			glBindBuffer(GL_ARRAY_BUFFER, vboPolygon);
			glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(vec4), &vertexData[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 1, 0);		// THERE WAS A ACCES EXCEPTION HERE!!!
			glDrawArrays(GL_LINE_LOOP, 0, wCps.size());
		}
	}
};

#pragma endregion


#pragma region curves

// Catmull-Rom
class CatmullRom {
	unsigned int vaoCurve, vboCurve;
	unsigned int vaoCtrlPoints, vboCtrlPoints;
	unsigned int vaoAnimatedObject, vboAnimatedObject;

	float tension = -1;		// tension
	std::vector<vec4> wCps;		// coordinates of control points
	std::vector<float> ts;	// parameters/knots

	// calculates the hermite curve between two given points
	vec4 Hermite(vec4 p0, vec4 v0, float t0, vec4 p1, vec4 v1, float t1, float t) {
		vec4 a0 = p0;
		vec4 a1 = v0;
		vec4 a2 = 3 * (p1 - p0) / (powf((t1 - t0), 2)) - (v1 + 2 * v0) / (t1 - t0);
		vec4 a3 = 2 * (p0 - p1) / (powf((t1 - t0), 3)) + (v1 + v0) / (powf((t1 - t0), 2));

		vec4 result = a3 * powf(t - t0, 3) + a2 * powf(t - t0, 2) + a1 * (t - t0) + a0;
		//printf("\tPoint: (%f, %f) at %f\n\n", result.x, result.y, t);
		return result;
	};
public:
	CatmullRom() {
		// Curve
		glGenVertexArrays(1, &vaoCurve);
		glBindVertexArray(vaoCurve);

		glGenBuffers(1, &vboCurve); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL); // attribute array, components/attribute, component type, normalize?, stride, offset

		// Control Points
		glGenVertexArrays(1, &vaoCtrlPoints);
		glBindVertexArray(vaoCtrlPoints);

		glGenBuffers(1, &vboCtrlPoints); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL); // attribute array, components/attribute, component type, normalize?, stride, offset

		// Animated Object
		glGenVertexArrays(1, &vaoAnimatedObject);
		glBindVertexArray(vaoAnimatedObject);

		glGenBuffers(1, &vboAnimatedObject); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vboAnimatedObject);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL); // attribute array, components/attribute, component type, normalize?, stride, offset

	}

	// Add a given 2D control point
	void AddControlPoint(float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		float ti = wCps.size();
		
		wCps.push_back(wVertex);
		ts.push_back(ti);
	}


	// Returns the selected control point or -1
	int PickControlPoint(float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		for (unsigned int p = 0; p < wCps.size(); p++) {
			if (dot(wCps[p] - wVertex, wCps[p] - wVertex) < 0.1) return p;
		}
		return -1;
	}

	// An indexed point is changed with new values
	void MoveControlPoint(int p, float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCps[p] = wVertex;
	}

	float tStart() { return ts[0]; }
	float tEnd() { return ts[wCps.size()-1]+1; }

	// calculates a point in a given time
	vec4 r(float t) {
		// searching for adjacent points
		for (int i = 0; i < wCps.size()+1; i++) {
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


			// checking whether the 2 knot points are adjacent
			if (ts[i] <= t && t <= ts2) {

				// we need previous, current, next and the point after the next
				vec4  p0 = wCps[prev], p1 = wCps[i], p2 = wCps[next], p3 = wCps[nextnext];
				float ts1 = ts[i];

				// we give a starting value for the velocity
				vec4 v1 = vec4(1, 1, 0, 0), v2 = vec4(1, 1, 0, 0);
				
				// if the current point is not the starting or the ending point, then we can calculate it
				v1 = (1 - tension) / 2 * ((p2 - p1) / (ts2 - ts1) + (p1 - p0) / (ts1 - ts0));
				v2 = (1 - tension) / 2 * ((p3 - p2) / (ts3 - ts2) + (p2 - p1) / (ts2 - ts1));


				// calculating a hermite curve between the current and the next point
				return Hermite(p1, v1, ts1, p2, v2, ts2, t);
			}
		}
	}

	// Draw everything
	void Draw() {
		mat4 VPTransform = camera.V() * camera.P();

		gpuProgram.setUniform(VPTransform, "MVP");

		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");

		if (wCps.size() > 0) {	// draw control points
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, wCps.size() * 4 * sizeof(float), &wCps[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);
			glPointSize(10.0f);
			glDrawArrays(GL_POINTS, 0, wCps.size());
		}

		if (wCps.size() >= 2) {	// draw curve
			std::vector<float> vertexData;
			for (int i = 0; i < nTesselatedVertices; i++) {	// Tessellate
				float tNormalized = (float)i / (nTesselatedVertices - 1);
				float t = tStart() + (tEnd() - tStart()) * tNormalized;
				//
				vec4 wVertex = r(t);

				// pushing data into the array
				vertexData.push_back(wVertex.x);
				vertexData.push_back(wVertex.y);
			}
			// copy data to the GPU
			glBindVertexArray(vaoCurve);
			glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
			glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 1, 0);
			glDrawArrays(GL_LINE_STRIP, 0, nTesselatedVertices);
		}
	}
};

#pragma endregion


// The virtual world: collection of two objects
CatmullRom* curve;
SimplePolygon* poly;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f);

	curve = new CatmullRom();
	poly = new SimplePolygon();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	curve->Draw();
	//poly->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') {
		//poly->Simplify();
	}

	glutPostRedisplay();        // redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }


int pickedControlPoint = -1;

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;

		curve->AddControlPoint(cX, cY);
		//poly->AddControlPoint(cX, cY);
		glutPostRedisplay();     // redraw
	}

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;

		pickedControlPoint = curve->PickControlPoint(cX, cY);
		//pickedControlPoint = poly->PickControlPoint(cX, cY);
		
		glutPostRedisplay();     // redraw
	}

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		pickedControlPoint = -1;
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if (pickedControlPoint >= 0) curve->MoveControlPoint(pickedControlPoint, cX, cY);
	//if (pickedControlPoint >= 0) poly->MoveControlPoint(pickedControlPoint, cX, cY);
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	glutPostRedisplay();					// redraw the scene
}