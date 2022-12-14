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

template<class T> struct Dnum {
	float f;
	T d; 
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 40;
bool textured = true;
float rotAngle = 0;

struct Camera {
	vec3 wEye, wLookat, wVup;		
	float fov, asp, fp, bp;
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;	
		fp = 1;	
		bp = 20;
	}

	mat4 V() { 
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(
				u.x,	v.x,	w.x,	0,
				u.y,	v.y,	w.y,	0,
				u.z,	v.z,	w.z,	0,
				0,		0,		0,		1
			);
	}

	mat4 P() { 
		return mat4(
			1 / (tan(fov / 2) * asp),	0,					0,							0,
			0,							1 / tan(fov / 2),	0,							0,
			0,							0,					-(fp + bp) / (bp - fp),		-1,
			0,							0,					-2 * fp * bp / (bp - fp),	0
		);
	}
};

struct Material {
	vec3 kd, ks, ka;
	float shininess;
};

struct Light {
	vec3 La, Le;
	vec4 wLightPos;
};

class CheckerBoardTexture : public Texture {
public:
	CheckerBoardTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		create(width, height, image, GL_NEAREST);
	}
};

struct RenderState {
	mat4				MVP, M, Minv, V, P; 
	Material*			material;
	std::vector<Light>	lights;
	Texture*			texture;
	vec3				wEye;
};

class Shader : public GPUProgram {
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

class PhongShader : public Shader {
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv;
		uniform Light[8] lights;
		uniform int   nLights;	
		uniform vec3  wEye; 

		uniform float rotAngle;

		layout(location = 0) in vec4  vtxPos; 
		layout(location = 1) in vec4  vtxNorm; 
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[8];	
		out vec2 texcoord;

		uniform bool textured;
		out float depthCue;

		void main() {
			float d = (vtxPos.x * vtxPos.x + vtxPos.y * vtxPos.y + vtxPos.z * vtxPos.z + vtxPos.w * vtxPos.w) / 2;
			
			vec4 rotatedXZ = vec4(vtxPos.x * cos(rotAngle) - vtxPos.z * sin(rotAngle), vtxPos.y, vtxPos.z * cos(rotAngle) + vtxPos.x * sin(rotAngle), vtxPos.w);
			vec4 rotatedYW = vec4(rotatedXZ.x, rotatedXZ.y * cos(rotAngle) - rotatedXZ.w * sin(rotAngle), rotatedXZ.z, rotatedXZ.w * cos(rotAngle) + rotatedXZ.y * sin(rotAngle));
			
			gl_Position = vec4(rotatedYW.xy, rotatedYW.z * d, 1) * MVP;

			vec4 translatedVtx = rotatedYW * M;
			depthCue = 1 / (dot(translatedVtx, translatedVtx) - 0.1f);
			
			vec4 wPos = vec4(vtxPos.xyz, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
			wView  = wEye * wPos.w - wPos.xyz;
			wNormal = (Minv * vec4(vtxNorm.xyz, 0)).xyz;
			texcoord = vtxUV;
		}
	)";

	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       
		in  vec3 wView;       
		in  vec3 wLight[8];  
		in  vec2 texcoord;
		
		uniform bool textured;
		in	float depthCue;

        out vec4 fragmentColor;

		void main() {
			if(textured){

				vec3 N = normalize(wNormal);
				vec3 V = normalize(wView); 
				if (dot(N, V) < 0) N = -N;	
				vec3 texColor = texture(diffuseTexture, texcoord).rgb;
				vec3 ka = material.ka * texColor;
				vec3 kd = material.kd * texColor;

				vec3 radiance = vec3(0, 0, 0);
				for(int i = 0; i < nLights; i++) {
					vec3 L = normalize(wLight[i]);
					vec3 H = normalize(L + V);
					float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);

					radiance += ka * lights[i].La + 
							   (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
				}
				fragmentColor = vec4(radiance, 1);

			} else {

				fragmentColor = vec4(depthCue, depthCue, depthCue, 1);

			}
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }
	
	void Bind(RenderState state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform(textured, "textured");

		setUniform(rotAngle, "rotAngle");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

class Geometry {
protected:
	unsigned int vao, vbo; 
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); 
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class ParamSurface : public Geometry {
	struct VertexData {
		vec4 position;
		vec4 normal;
		vec2 texcoord;
	};

	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }
	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z, Dnum2& W) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;

		vtxData.texcoord = vec2(u, v);	

		Dnum2 X, Y, Z, W;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z, W);		

		vtxData.position = vec4(X.f, Y.f, Z.f, W.f);	

		vec4 drdU(X.d.x, Y.d.x, Z.d.x, W.d.x),	
			 drdV(X.d.y, Y.d.y, Z.d.y, W.d.y);	
		vtxData.normal = vec4(
			drdU.y * drdV.z - drdU.z * drdV.y,
			drdU.z * drdV.w - drdU.w * drdV.z,
			drdU.w * drdV.x - drdU.x * drdV.w,
			drdU.x * drdV.y - drdU.y * drdV.x
		);

		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0); 
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2); 

		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		if (textured) {
			for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
		}
		else {
			for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_LINES, i * nVtxPerStrip, nVtxPerStrip);
		}
	}
};

class Klein : public ParamSurface {
	const float size = 1.5f;
public:
	Klein() { create(); }

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z, Dnum2& W) {
		U = U * M_PI * 2;
		V = V * M_PI * 2;
		float epsilon = 0.1;
		float P = 0.5, R = 0.5;

		Dnum2 c = (Sin(V) * epsilon + 1.0f) * P;
		X = (Cos(U / 2) * Cos(V) - Sin(U / 2) * Sin(V * 2)) * R;
		Y = (Sin(U / 2) * Cos(V) + Cos(U / 2) * Sin(V * 2)) * R;
		Z = c * Cos(U);
		W = c * Sin(U);
	}
};

struct Object {
	Shader* shader;
	Material* material;
	Texture* texture;
	Geometry* geometry;
	vec3 scale;
public:
	Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		scale(vec3(5, 5, 5)) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale);
		
		Minv = ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);

		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;

		state.material = material;
		state.texture = texture;
		shader->Bind(state);	
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { 
		rotAngle = 0.8f * tend;
	}
};

class Scene {
	std::vector<Object*> objects;
	Camera camera;
	std::vector<Light> lights;
public:
	void Build() {
		Shader* phongShader = new PhongShader();

		Material* material1 = new Material;
		material1->kd = vec3(0.8f, 0.6f, 0.4f);
		material1->ks = vec3(0.3f, 0.3f, 0.3f);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 30;

		Texture* texture4x8 = new CheckerBoardTexture(4, 8);
		Texture* texture15x20 = new CheckerBoardTexture(15, 20);

		Geometry* klein = new Klein();

		Object* kleinObject = new Object(phongShader, material1, texture4x8, klein);
		objects.push_back(kleinObject);

		camera.wEye = vec3(0, 0, 8);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		lights.resize(3);
		lights[0].wLightPos = vec4(5, 5, 4, 0);
		lights[0].La = vec3(0.1f, 0.1f, 1);
		lights[0].Le = vec3(3, 0, 0);

		lights[1].wLightPos = vec4(5, 10, 20, 0);
		lights[1].La = vec3(0.2f, 0.2f, 0.2f);
		lights[1].Le = vec3(0, 3, 0);

		lights[2].wLightPos = vec4(-5, 5, 5, 0);
		lights[2].La = vec3(0.1f, 0.1f, 0.1f);
		lights[2].Le = vec3(0, 0, 3);
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object* obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		for (Object* obj : objects) obj->Animate(tstart, tend);
	}
};

Scene scene;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
	scene.Render();
	glutSwapBuffers();	
}

void onKeyboard(unsigned char key, int pX, int pY) { 
	if (key == ' ') {
		textured = !textured;
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY) { }

void onMouseMotion(int pX, int pY) { }

void onIdle() {
	static float tend = 0;
	const float dt = 0.1f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}