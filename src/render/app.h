#ifndef _GL_MAIN_HEADER_H
#define _GL_MAIN_HEADER_H

#define VNAME		4DF
#define VTYPE		float

class Vector4DF {
public:
    VTYPE x, y, z, w;

    Vector4DF &Set(const float xa, const float ya, const float za)	{ x = xa; y = ya; z = za; w = 1; return *this; }
    Vector4DF &Set(const float xa, const float ya, const float za, const float wa)	{ x = xa; y = ya; z = za; w = wa; return *this; }

    // Constructors/Destructors
    Vector4DF() { x = 0; y = 0; z = 0; w = 0; }
    Vector4DF(const VTYPE xa, const VTYPE ya, const VTYPE za, const VTYPE wa);

    Vector4DF(const Vector4DF &op);

    // Member Functions
    Vector4DF &operator= (const int op);
    Vector4DF &operator= (const double op);

    Vector4DF &operator= (const Vector4DF &op);

    Vector4DF &operator+= (const int op);
    Vector4DF &operator+= (const float op);
    Vector4DF &operator+= (const double op);

    Vector4DF &operator+= (const Vector4DF &op);

    Vector4DF &operator-= (const int op);
    Vector4DF &operator-= (const double op);

    Vector4DF &operator-= (const Vector4DF &op);

    Vector4DF &operator*= (const int op);
    Vector4DF &operator*= (const double op);

    Vector4DF &operator*= (const Vector4DF &op);
    Vector4DF &operator*= (const float* op);

    Vector4DF &operator/= (const int op);
    Vector4DF &operator/= (const double op);

    // Slow operations - require temporary variables
    Vector4DF operator+ (const int op)			{ return Vector4DF(x + float(op), y + float(op), z + float(op), w + float(op)); }
    Vector4DF operator+ (const float op)		{ return Vector4DF(x + op, y + op, z + op, w*op); }
    Vector4DF operator+ (const Vector4DF &op)	{ return Vector4DF(x + op.x, y + op.y, z + op.z, w + op.w); }
    Vector4DF operator- (const int op)			{ return Vector4DF(x - float(op), y - float(op), z - float(op), w - float(op)); }
    Vector4DF operator- (const float op)		{ return Vector4DF(x - op, y - op, z - op, w*op); }
    Vector4DF operator- (const Vector4DF &op)	{ return Vector4DF(x - op.x, y - op.y, z - op.z, w - op.w); }
    Vector4DF operator* (const int op)			{ return Vector4DF(x*float(op), y*float(op), z*float(op), w*float(op)); }
    Vector4DF operator* (const float op)		{ return Vector4DF(x*op, y*op, z*op, w*op); }
    Vector4DF operator* (const Vector4DF &op)	{ return Vector4DF(x*op.x, y*op.y, z*op.z, w*op.w); }
    // --

    Vector4DF& Clamp(float xc, float yc, float zc, float wc)
    {
        x = (x > xc) ? xc : x;
        y = (y > yc) ? yc : y;
        z = (z > zc) ? zc : z;
        w = (w > wc) ? wc : w;
        return *this;
    }

    Vector4DF &Cross(const Vector4DF &v);

    double Dot(const Vector4DF &v);

    double Dist(const Vector4DF &v);

    double DistSq(const Vector4DF &v);

    Vector4DF &Normalize(void);
    double Length(void);

    VTYPE &X(void)				{ return x; }
    VTYPE &Y(void)				{ return y; }
    VTYPE &Z(void)				{ return z; }
    VTYPE &W(void)				{ return w; }
    const VTYPE &X(void) const	{ return x; }
    const VTYPE &Y(void) const	{ return y; }
    const VTYPE &Z(void) const	{ return z; }
    const VTYPE &W(void) const	{ return w; }
    VTYPE *Data(void)			{ return &x; }
};

#undef VNAME
#undef VTYPE

#endif/*_GL_MAIN_HEADER_H*/
