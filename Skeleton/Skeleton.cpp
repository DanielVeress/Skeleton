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
	std::vector<vec4> wCtrlPoints;

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
		wCtrlPoints.push_back(wVertex);
	}

	// Returns the selected control point or -1
	int PickControlPoint(float cX, float cY) {
		vec4 mouse = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		for (unsigned int p = 0; p < wCtrlPoints.size(); p++) {
			if (dot(wCtrlPoints[p] - mouse, wCtrlPoints[p] - mouse) < 0.1) 
				return p;
		}


		if (wCtrlPoints.size() < 2) {
			return -1;
		}

		float bestDistance = -9999;
		int bestIdx = 0;
		// for every line we calculate the distance
		for (int i = 0; i < wCtrlPoints.size() - 1; i++) {

			vec4 lineVec = wCtrlPoints[i + 1] - wCtrlPoints[i]; // we get the vector that points from i to i+1
			vec4 mouseVec = mouse - wCtrlPoints[i]; // we get vector that points from i to mouse point

			float shadowLength = (dot(mouseVec, lineVec) / dot(lineVec, lineVec));
			if (shadowLength < 0)
				shadowLength = 0;
			else if (shadowLength > 1)
				shadowLength = 1;

			vec4 closestPointToMouse = wCtrlPoints[i] + (lineVec * shadowLength);
			float distance = dot(closestPointToMouse - mouse, closestPointToMouse - mouse);
			if (bestDistance < distance) {
				bestDistance = distance;
				bestIdx = i+1;
			}
		}

		wCtrlPoints.insert(wCtrlPoints.begin() + bestIdx, mouse);

		return bestIdx;
	}

	// An indexed point is changed with new values
	void MoveControlPoint(int p, float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCtrlPoints[p] = wVertex;
	}

	// Draw everything
	void Draw() {
		mat4 VPTransform = camera.V() * camera.P();

		gpuProgram.setUniform(VPTransform, "MVP");

		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");

		// draw ctrl points
		if (wCtrlPoints.size() > 0) {	
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, wCtrlPoints.size() * 4 * sizeof(float), &wCtrlPoints[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);
			glPointSize(10.0f);
			glDrawArrays(GL_POINTS, 0, wCtrlPoints.size());
		}

		// draw plygon
		if (wCtrlPoints.size() >= 2) {
			std::vector<float> vertexData;
			for (unsigned int i = 0; i < wCtrlPoints.size(); i++) {
				vertexData.push_back(wCtrlPoints[i].x);
				vertexData.push_back(wCtrlPoints[i].y);
			}

			// copy data to the GPU
			glBindVertexArray(vaoPolygon);
			glBindBuffer(GL_ARRAY_BUFFER, vboPolygon);
			glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(vec4), &vertexData[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 1, 0);		// THERE WAS A ACCES EXCEPTION HERE!!!
			glDrawArrays(GL_LINE_LOOP, 0, wCtrlPoints.size());
		}
	}
};

#pragma endregion


#pragma region curves

class Curve {
	unsigned int vaoCurve, vboCurve;
	unsigned int vaoCtrlPoints, vboCtrlPoints;
	unsigned int vaoAnimatedObject, vboAnimatedObject;
protected:
	std::vector<vec4> wCtrlPoints;		// coordinates of control points
public:
	Curve() {
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

	virtual vec4 r(float t) { return wCtrlPoints[0]; }
	virtual float tStart() { return 0; }
	virtual float tEnd() { return 1; }

	// Add a given 2D control point
	virtual void AddControlPoint(float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCtrlPoints.push_back(wVertex);
	}

	// Returns the selected control point or -1
	int PickControlPoint(float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		for (unsigned int p = 0; p < wCtrlPoints.size(); p++) {
			if (dot(wCtrlPoints[p] - wVertex, wCtrlPoints[p] - wVertex) < 0.1) return p;
		}
		return -1;
	}

	// getting closest point on a line to a point
	int GetClosestPointOnLine(float cX, float cY) {
		if (wCtrlPoints.size() < 2) {
												// TODO test this part
			AddControlPoint(cX, cY);
			return wCtrlPoints.size()-1;
		}

		vec4 mouse = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();

		vec4 bestVec = vec4(9999, 9999, 0, 0);
		int bestIdx = 0;
		// for every line we calculate the distance
		for (int i = 0; i < wCtrlPoints.size()-1; i++) {
			
			vec4 lineVec = wCtrlPoints[i + 1] - wCtrlPoints[i]; // we get the vector that points from i to i+1
			vec4 mouseVec = mouse - wCtrlPoints[i]; // we get vector that points from i to mouse point

			float shadowLength = (dot(mouseVec, lineVec) / dot(lineVec, lineVec));
			if (shadowLength < 0)
				shadowLength = 0;
			else if (shadowLength > 1)
				shadowLength = 1;

			vec4 vecToResult = wCtrlPoints[i] + (lineVec * shadowLength);
			if (vecToResult.x * vecToResult.x + vecToResult.y * vecToResult.y < bestVec.x * bestVec.x + bestVec.y * bestVec.y) {
				bestVec = vecToResult;
				bestIdx = i;
			}
		}

		wCtrlPoints.insert(wCtrlPoints.begin() + bestIdx, mouse);
		return bestIdx;
	}

	// An indexed point is changed with new values
	void MoveControlPoint(int p, float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCtrlPoints[p] = wVertex;
	}

	// Draw everything
	void Draw() {
		mat4 VPTransform = camera.V() * camera.P();

		gpuProgram.setUniform(VPTransform, "MVP");

		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");

		if (wCtrlPoints.size() > 0) {	// draw control points
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, wCtrlPoints.size() * 4 * sizeof(float), &wCtrlPoints[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);
			glPointSize(10.0f);
			glDrawArrays(GL_POINTS, 0, wCtrlPoints.size());
		}

		if (wCtrlPoints.size() >= 2) {	// draw curve
			std::vector<float> vertexData;
			for (int i = 0; i < nTesselatedVertices; i++) {	// Tessellate
				float tNormalized = (float)i / (nTesselatedVertices - 1);
				float t = tStart() + (tEnd() - tStart()) * tNormalized;
				vec4 wVertex = r(t);
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

// Bezier using Bernstein polynomials
class BezierCurve : public Curve {
	float B(int i, float t) {
		int n = wCtrlPoints.size() - 1; // n deg polynomial = n+1 pts!
		float choose = 1;
		for (int j = 1; j <= i; j++) choose *= (float)(n - j + 1) / j;
		return choose * pow(t, i) * pow(1 - t, n - i);
	}
public:
	virtual vec4 r(float t) {
		vec4 wPoint = vec4(0, 0, 0, 0);
		for (unsigned int n = 0; n < wCtrlPoints.size(); n++) wPoint += wCtrlPoints[n] * B(n, t);
		return wPoint;
	}
};

// Lagrange curve
class LagrangeCurve : public Curve {
	std::vector<float> ts;  // knots
	float L(int i, float t) {
		float Li = 1.0f;
		for (unsigned int j = 0; j < wCtrlPoints.size(); j++)
			if (j != i) Li *= (t - ts[j]) / (ts[i] - ts[j]);
		return Li;
	}
public:
	void AddControlPoint(float cX, float cY) {
		ts.push_back((float)wCtrlPoints.size());
		Curve::AddControlPoint(cX, cY);
	}
	float tStart() { return ts[0]; }
	float tEnd() { return ts[wCtrlPoints.size() - 1]; }

	virtual vec4 r(float t) {
		vec4 wPoint = vec4(0, 0, 0, 0);
		for (unsigned int n = 0; n < wCtrlPoints.size(); n++) wPoint += wCtrlPoints[n] * L(n, t);
		return wPoint;
	}
};

#pragma endregion


// The virtual world: collection of two objects
Curve* curve;
SimplePolygon* poly;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f);

	//curve = new Curve();
	poly = new SimplePolygon();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	//curve->Draw();
	poly->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	glutPostRedisplay();        // redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}


int pickedControlPoint = -1;

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;

		//curve->AddControlPoint(cX, cY);
		poly->AddControlPoint(cX, cY);
		glutPostRedisplay();     // redraw
	}

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;

		//pickedControlPoint = curve->GetClosestPointOnLine(cX, cY);
		//pickedControlPoint = curve->PickControlPoint(cX, cY);
		pickedControlPoint = poly->PickControlPoint(cX, cY);
		
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
	//if (pickedControlPoint >= 0) curve->MoveControlPoint(pickedControlPoint, cX, cY);
	if (pickedControlPoint >= 0) poly->MoveControlPoint(pickedControlPoint, cX, cY);
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	glutPostRedisplay();					// redraw the scene
}