#include <GL\glew.h>
#include <GL\freeglut.h>
#include <sstream>
#include <cuda_runtime.h>
#include "save_screen.h"
#include "sph_timer.h"
#include "sph_data.h"
#include "sph_hybrid_system.h"
#include "gl_main_header.h"

//#pragma comment(lib, "glew32.lib") 

sph::HybridSystem *sph_system;

Timer *sph_timer;
char *window_title;

GLuint v;
GLuint f;
GLuint p;

Vector4DF	light[2], light_to[2];				// Light stuff

static const std::string scale_density_filename = "scale_d.txt";
static const std::string scale_force_filename = "scale_f.txt";

bool screenshot = false;

bool init_cuda(void)
{
    int count = 0;
    int i = 0;

    cudaGetDeviceCount(&count);
    if (count == 0)
    {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    for (i = 0; i < count; i++)
    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
        {
            if (prop.major >= 1)
            {
                break;
            }
        }
    }

    if (i == count)
    {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        return false;
    }

    cudaSetDevice(i);

    printf("CUDA initialized.\n");
    return true;
}

void set_shaders()
{
    char *vs = NULL;
    char *fs = NULL;

    vs = (char *)malloc(sizeof(char) * 10000);
    fs = (char *)malloc(sizeof(char) * 10000);
    memset(vs, 0, sizeof(char) * 10000);
    memset(fs, 0, sizeof(char) * 10000);

    FILE *fp;
    char c;
    int count;

    fp = fopen("shader/shader.vs", "r");
    count = 0;
    while ((c = fgetc(fp)) != EOF)
    {
        vs[count] = c;
        count++;
    }
    fclose(fp);

    fp = fopen("shader/shader.fs", "r");
    count = 0;
    while ((c = fgetc(fp)) != EOF)
    {
        fs[count] = c;
        count++;
    }
    fclose(fp);

    v = glCreateShader(GL_VERTEX_SHADER);
    f = glCreateShader(GL_FRAGMENT_SHADER);

    const char *vv;
    const char *ff;
    vv = vs;
    ff = fs;

    glShaderSource(v, 1, &vv, NULL);
    glShaderSource(f, 1, &ff, NULL);

    int success;

    glCompileShader(v);
    glGetShaderiv(v, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char info_log[5000];
        glGetShaderInfoLog(v, 5000, NULL, info_log);
        printf("Error in vertex shader compilation!\n");
        printf("Info Log: %s\n", info_log);
    }

    glCompileShader(f);
    glGetShaderiv(f, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char info_log[5000];
        glGetShaderInfoLog(f, 5000, NULL, info_log);
        printf("Error in fragment shader compilation!\n");
        printf("Info Log: %s\n", info_log);
    }

    p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glUseProgram(p);

    free(vs);
    free(fs);
}

void draw_box(float ox, float oy, float oz, float width, float height, float length)
{
    glLineWidth(2.5f);
    glColor3f(0.8f, 0.8f, 0.8f);

    glBegin(GL_LINES);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox + width, oy, oz);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox, oy, oz + length);

    glVertex3f(ox + width, oy, oz);
    glVertex3f(ox + width, oy + height, oz);

    glVertex3f(ox + width, oy + height, oz);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox, oy, oz + length);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox + width, oy, oz);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox, oy, oz + length);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox + width, oy + height, oz);
    glVertex3f(ox + width, oy + height, oz + length);

    glVertex3f(ox + width, oy + height, oz + length);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox + width, oy + height, oz + length);

    glEnd();
}

//sf 设置长方体
void init_sph_system()
{
    real_world_origin.x = -40.0f;
    real_world_origin.y = -40.0f;
    real_world_origin.z = -40.0f;

    real_world_side.x = 80.0f;
    real_world_side.y = 80.0f;
    real_world_side.z = 80.0f;

    sph_system = new sph::HybridSystem(real_world_side, real_world_origin);

    sph_timer = new Timer();
    window_title = (char *)malloc(sizeof(char) * 50);
}

void init()
{
    glewInit();

    glViewport(0, 0, window_width, window_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(45.0, (float)window_width / window_height, 10.0f, 500.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -3.0f);

    light[0].x = 0;		light[0].y = 55;	light[0].z = 0; light[0].w = 1;
    light_to[0].x = 0;	light_to[0].y = 0;	light_to[0].z = 0; light_to[0].w = 1;

    light[1].x = 55;		light[1].y = 140;	light[1].z = 50;	light[1].w = 1;
    light_to[1].x = 0;	light_to[1].y = 0;	light_to[1].z = 0;		light_to[1].w = 1;
}

void init_ratio()
{
    //sim_ratio = real_world_side / sph->hParam->world_size;
}

void draw_scene()
{
    glPushMatrix();

    // set camera
    xRot += (xRotLength - xRot)*0.1f;
    yRot += (yRotLength - yRot)*0.1f;
    glTranslatef(xTrans, yTrans, zTrans);
	
    glRotatef(xRot, 1.0f, 0.0f, 0.0f);
    glRotatef(yRot, 0.0f, 1.0f, 0.0f);
	//glTranslatef(0, 11.7, -154.2);
	//glRotatef(25.2, 1.0f, 0.0f, 0.0f);
	//glRotatef(136.4, 0.0f, 1.0f, 0.0f);
    // draw framework
    glDisable(GL_LIGHTING);
    //draw_box(real_world_origin.x, real_world_origin.y, real_world_origin.z, real_world_side.x, real_world_side.y, real_world_side.z);
	//std::cout << xTrans << "  " << yTrans << "  "<<zTrans<<"------"<<xRot << "      " << yRot << std::endl;
    //draw light
    draw_box(light[0].x - 0.2f, light[0].y - 0.2f, light[0].z - 0.2f, 0.4f, 0.4f, 0.4f);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glDisable(GL_COLOR_MATERIAL);

    Vector4DF amb, diff, spec;
    float shininess = 5.0;

    float pos[4];
    pos[0] = light[0].x;
    pos[1] = light[0].y;
    pos[2] = light[0].z;
    pos[3] = 1;
    amb.Set(0, 0, 0, 1); diff.Set(1, 1, 1, 1); spec.Set(1, 1, 1, 1);
    glLightfv(GL_LIGHT0, GL_POSITION, (float*)&pos[0]);
    glLightfv(GL_LIGHT0, GL_AMBIENT, (float*)&amb.x);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (float*)&diff.x);
    glLightfv(GL_LIGHT0, GL_SPECULAR, (float*)&spec.x);

    //GLfloat spot_cutoff = 70.0f;
    //GLfloat spot_pos[] = {0, -1.0, 0};
    //glLightfv(GL_LIGHT0, GL_SPOT_CUTOFF, &spot_cutoff);
    //glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, spot_pos);

    amb.Set(0, 0, 0, 1); diff.Set(.3, .3, .3, 1); spec.Set(.1, .1, .1, 1);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (float*)&amb.x);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (float*)&diff.x);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (float*)&spec.x);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, (float*)&shininess);

    sph_system->drawParticles(0.8f, psize);
    sph_system->drawInfo(window_width, window_height);

    glPopMatrix();
}

void display_func()
{
    sph_system->tick();

    //glEnable(GL_POINT_SMOOTH);
    glEnable(GL_DEPTH_TEST);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);   //sf 背景颜色
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_CULL_FACE);
    glShadeModel(GL_SMOOTH);
    glMatrixMode(GL_MODELVIEW);

    draw_scene();

    glutSwapBuffers();

    sph_timer->update();
    memset(window_title, 0, 50);
    sprintf(window_title, "Hybrid GPU Parallel SPH. FPS: %f", sph_timer->get_fps());
    glutSetWindowTitle(window_title);


	if (screenshot)
	{
		static unsigned int step = 0;

		if (step % 8 == 0)
		{
			std::stringstream ss;
			ss << "screenshot/step_";
			ss.fill('0');
			ss.width(5);
			ss << step / 8;
			std::string file_path = ss.str();
			SaveScreenShot(window_width, window_height, file_path);
		}

		++step;
	}
    //if (screenshot)
    //{
    //    //static unsigned int step = 0;

    //    /*std::stringstream ss;
    //    ss << "screenshot/step_";
    //    ss.fill('0');
    //    ss.width(5);
    //    ss << sph_system->loop;
    //    std::string file_path = ss.str();*/

    //    char bmp_name[30] = "screenshot/step_";
    //    char number[10];
    //    //        itoa(sph_system->loop, number, 10);

    //    strcat(bmp_name, number);
    //    SaveScreenShot(window_width, window_height, bmp_name);

    //    //++step;
    //}
}

void idle_func()
{
    glutPostRedisplay();
}

void reshape_func(GLint width, GLint height)
{
    window_width = width;
    window_height = height;

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(45.0, (float)width / height, 0.001, 500.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -3.0f);
}

void keyboard_func(unsigned char key, int x, int y)
{
    if (key == 't')
    {
        sph_system->tick();

        //for (size_t i = 0; i < 4; ++i)
        //{
        //    float3 pos = sph->get_position(i);
        //    printf("(%f, %f, %f) \n", pos.x, pos.y, pos.z);
        //}
        //printf("\n");
        printf("--------\n");
    }

    if (key == ' ')
    {
        sph_system->setPause();
    }

    if (key == 'w')
    {
        zTrans += 1.3f;
    }

	if (key == 'o')
	{
		psize += 1;
	}
	if (key == 'u')
	{
		psize -= 1;
	}

    if (key == 's')
    {
        zTrans -= 1.3f;
    }

    if (key == 'a')
    {
        xTrans += 1.3f;
    }

    if (key == 'd')
    {
        xTrans -= 1.3f;
    }

    if (key == 'q')
    {
        yTrans -= 1.3f;
    }

    if (key == 'e')
    {
        yTrans += 1.3f;
    }

    if (key == '1')
    {
        sph_system->insertParticles(1);
    }

    if (key == '/')
    {
        screenshot = !screenshot;
    }

    glutPostRedisplay();
}

void special_keyboard_func(int key, int x, int y)
{
    switch (key)
    {
    case GLUT_KEY_UP:
        light[0].z -= 0.1f;
        break;
    case GLUT_KEY_DOWN:
        light[0].z += 0.1f;
        break;
    case GLUT_KEY_LEFT:
        light[0].x -= 0.1f;
        break;
    case GLUT_KEY_RIGHT:
        light[0].x += 0.1f;
        break;
    case GLUT_KEY_PAGE_UP:
        light[0].y += 0.1f;
        break;
    case GLUT_KEY_PAGE_DOWN:
        light[0].y -= 0.1f;
        break;
    default:
        break;
    }

    printf("light pos: %f, %f, %f\n", light[0].x, light[0].y, light[0].z);

    glutPostRedisplay();
}

void mouse_func(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        buttonState = 1;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x; oy = y;

    glutPostRedisplay();
}

void motion_func(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 1)
    {
        xRotLength += dy / 5.0f;
        yRotLength += dx / 5.0f;
    }

    ox = x; oy = y;

    glutPostRedisplay();
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("SPH Fluid 3D");

    if (!init_cuda()) return -1;
    init();
    init_sph_system();
    init_ratio();
    //set_shaders();
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
    glEnable(GL_POINT_SPRITE_ARB);
    glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    glutDisplayFunc(display_func);
    glutReshapeFunc(reshape_func);
    glutKeyboardFunc(keyboard_func);
    glutSpecialFunc(special_keyboard_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutIdleFunc(idle_func);


    glutMainLoop();

    return 0;
}
