// ConsoleApplication1.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//

#pragma once

#include <cmath>
#include <iostream>
#include <initializer_list>

namespace EngineMath {

    
    const float PI = 3.14159265358979323846f;
    const float PI_2 = PI / 2.0f;
    const float PI_4 = PI / 4.0f;
    const float TWO_PI = 2.0f * PI;
    const float EPSILON = 1e-6f;
    const float DEG_TO_RAD = PI / 180.0f;
    const float RAD_TO_DEG = 180.0f / PI;


    inline float sqrt(float x) { return std::sqrt(x); }
    inline float square(float x) { return x * x; }
    inline float cube(float x) { return x * x * x; }
    inline float power(float base, float exp) { return std::pow(base, exp); }
    inline float abs(float x) { return std::abs(x); }
    inline float EMax(float a, float b) { return (a > b) ? a : b; }
    inline float EMin(float a, float b) { return (a < b) ? a : b; }
    inline float round(float x) { return std::round(x); }
    inline float floor(float x) { return std::floor(x); }
    inline float ceil(float x) { return std::ceil(x); }
    inline float fabs(float x) { return std::fabs(x); }
    inline float mod(float x, float y) { return std::fmod(x, y); }
    inline float exp(float x) { return std::exp(x); }
    inline float log(float x) { return std::log(x); }
    inline float log10(float x) { return std::log10(x); }


    inline float sin(float x) { return std::sin(x); }
    inline float cos(float x) { return std::cos(x); }
    inline float tan(float x) { return std::tan(x); }
    inline float asin(float x) { return std::asin(x); }
    inline float acos(float x) { return std::acos(x); }
    inline float atan(float x) { return std::atan(x); }
    inline float sinh(float x) { return std::sinh(x); }
    inline float cosh(float x) { return std::cosh(x); }
    inline float tanh(float x) { return std::tanh(x); }


    inline float radians(float degrees) { return degrees * DEG_TO_RAD; }
    inline float degrees(float radians) { return radians * RAD_TO_DEG; }


    inline float circleArea(float radius) { return PI * radius * radius; }
    inline float circleCircumference(float radius) { return TWO_PI * radius; }
    inline float rectangleArea(float width, float height) { return width * height; }
    inline float rectanglePerimeter(float width, float height) { return 2.0f * (width + height); }
    inline float triangleArea(float base, float height) { return 0.5f * base * height; }
    inline float distance(float x1, float y1, float x2, float y2) {
        return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    }


    inline float lerp(float a, float b, float t) { return a + t * (b - a); }
    inline float factorial(int n) {
        if (n <= 1) return 1.0f;
        float result = 1.0f;
        for (int i = 2; i <= n; ++i) {
            result *= i;
        }
        return result;
    }
    inline bool approxEqual(float a, float b, float epsilon = EPSILON) {
        return abs(a - b) < epsilon;
    }



    struct Vector2 {
        float x, y;


        Vector2() : x(0.0f), y(0.0f) {}
        Vector2(float x, float y) : x(x), y(y) {}
        Vector2(const Vector2& v) : x(v.x), y(v.y) {}


        Vector2& operator=(const Vector2& v) {
            x = v.x; y = v.y;
            return *this;
        }


        Vector2 operator+(const Vector2& v) const { return Vector2(x + v.x, y + v.y); }
        Vector2 operator-(const Vector2& v) const { return Vector2(x - v.x, y - v.y); }
        Vector2 operator*(float scalar) const { return Vector2(x * scalar, y * scalar); }
        Vector2 operator/(float scalar) const { return Vector2(x / scalar, y / scalar); }
        Vector2 operator-() const { return Vector2(-x, -y); }

  
        Vector2& operator+=(const Vector2& v) { x += v.x; y += v.y; return *this; }
        Vector2& operator-=(const Vector2& v) { x -= v.x; y -= v.y; return *this; }
        Vector2& operator*=(float scalar) { x *= scalar; y *= scalar; return *this; }
        Vector2& operator/=(float scalar) { x /= scalar; y /= scalar; return *this; }


        bool operator==(const Vector2& v) const { return approxEqual(x, v.x) && approxEqual(y, v.y); }
        bool operator!=(const Vector2& v) const { return !(*this == v); }


        float magnitude() const { return sqrt(x * x + y * y); }
        float magnitudeSquared() const { return x * x + y * y; }
        Vector2 normalized() const {
            float mag = magnitude();
            return mag > EPSILON ? *this / mag : Vector2(0, 0);
        }
        void normalize() { *this = normalized(); }
        float dot(const Vector2& v) const { return x * v.x + y * v.y; }
        float cross(const Vector2& v) const { return x * v.y - y * v.x; }
        Vector2 lerp(const Vector2& v, float t) const { return *this + (v - *this) * t; }


        static Vector2 zero() { return Vector2(0, 0); }
        static Vector2 one() { return Vector2(1, 1); }
        static Vector2 up() { return Vector2(0, 1); }
        static Vector2 down() { return Vector2(0, -1); }
        static Vector2 left() { return Vector2(-1, 0); }
        static Vector2 right() { return Vector2(1, 0); }
    };

    struct Vector3 {
        float x, y, z;


        Vector3() : x(0.0f), y(0.0f), z(0.0f) {}
        Vector3(float x, float y, float z) : x(x), y(y), z(z) {}
        Vector3(const Vector3& v) : x(v.x), y(v.y), z(v.z) {}

        Vector3& operator=(const Vector3& v) {
            x = v.x; y = v.y; z = v.z;
            return *this;
        }

        Vector3 operator+(const Vector3& v) const { return Vector3(x + v.x, y + v.y, z + v.z); }
        Vector3 operator-(const Vector3& v) const { return Vector3(x - v.x, y - v.y, z - v.z); }
        Vector3 operator*(float scalar) const { return Vector3(x * scalar, y * scalar, z * scalar); }
        Vector3 operator/(float scalar) const { return Vector3(x / scalar, y / scalar, z / scalar); }
        Vector3 operator-() const { return Vector3(-x, -y, -z); }

        Vector3& operator+=(const Vector3& v) { x += v.x; y += v.y; z += v.z; return *this; }
        Vector3& operator-=(const Vector3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
        Vector3& operator*=(float scalar) { x *= scalar; y *= scalar; z *= scalar; return *this; }
        Vector3& operator/=(float scalar) { x /= scalar; y /= scalar; z /= scalar; return *this; }

        bool operator==(const Vector3& v) const {
            return approxEqual(x, v.x) && approxEqual(y, v.y) && approxEqual(z, v.z);
        }
        bool operator!=(const Vector3& v) const { return !(*this == v); }

        float magnitude() const { return sqrt(x * x + y * y + z * z); }
        float magnitudeSquared() const { return x * x + y * y + z * z; }
        Vector3 normalized() const {
            float mag = magnitude();
            return mag > EPSILON ? *this / mag : Vector3(0, 0, 0);
        }
        void normalize() { *this = normalized(); }
        float dot(const Vector3& v) const { return x * v.x + y * v.y + z * v.z; }
        Vector3 cross(const Vector3& v) const {
            return Vector3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
        }
        Vector3 lerp(const Vector3& v, float t) const { return *this + (v - *this) * t; }

        static Vector3 zero() { return Vector3(0, 0, 0); }
        static Vector3 one() { return Vector3(1, 1, 1); }
        static Vector3 up() { return Vector3(0, 1, 0); }
        static Vector3 down() { return Vector3(0, -1, 0); }
        static Vector3 left() { return Vector3(-1, 0, 0); }
        static Vector3 right() { return Vector3(1, 0, 0); }
        static Vector3 forward() { return Vector3(0, 0, 1); }
        static Vector3 back() { return Vector3(0, 0, -1); }
    };

    struct Vector4 {
        float x, y, z, w;

        Vector4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
        Vector4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
        Vector4(const Vector3& v, float w) : x(v.x), y(v.y), z(v.z), w(w) {}
        Vector4(const Vector4& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

        Vector4& operator=(const Vector4& v) {
            x = v.x; y = v.y; z = v.z; w = v.w;
            return *this;
        }

        Vector4 operator+(const Vector4& v) const { return Vector4(x + v.x, y + v.y, z + v.z, w + v.w); }
        Vector4 operator-(const Vector4& v) const { return Vector4(x - v.x, y - v.y, z - v.z, w - v.w); }
        Vector4 operator*(float scalar) const { return Vector4(x * scalar, y * scalar, z * scalar, w * scalar); }
        Vector4 operator/(float scalar) const { return Vector4(x / scalar, y / scalar, z / scalar, w / scalar); }
        Vector4 operator-() const { return Vector4(-x, -y, -z, -w); }

        Vector4& operator+=(const Vector4& v) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
        Vector4& operator-=(const Vector4& v) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *this; }
        Vector4& operator*=(float scalar) { x *= scalar; y *= scalar; z *= scalar; w *= scalar; return *this; }
        Vector4& operator/=(float scalar) { x /= scalar; y /= scalar; z /= scalar; w /= scalar; return *this; }

        bool operator==(const Vector4& v) const {
            return approxEqual(x, v.x) && approxEqual(y, v.y) && approxEqual(z, v.z) && approxEqual(w, v.w);
        }
        bool operator!=(const Vector4& v) const { return !(*this == v); }

        float magnitude() const { return sqrt(x * x + y * y + z * z + w * w); }
        float magnitudeSquared() const { return x * x + y * y + z * z + w * w; }
        Vector4 normalized() const {
            float mag = magnitude();
            return mag > EPSILON ? *this / mag : Vector4(0, 0, 0, 0);
        }
        void normalize() { *this = normalized(); }
        float dot(const Vector4& v) const { return x * v.x + y * v.y + z * v.z + w * v.w; }
        Vector4 lerp(const Vector4& v, float t) const { return *this + (v - *this) * t; }

        Vector3 xyz() const { return Vector3(x, y, z); }
    };


    struct Quaternion {
        float x, y, z, w;

        Quaternion() : x(0.0f), y(0.0f), z(0.0f), w(1.0f) {}
        Quaternion(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
        Quaternion(const Quaternion& q) : x(q.x), y(q.y), z(q.z), w(q.w) {}

        Quaternion& operator=(const Quaternion& q) {
            x = q.x; y = q.y; z = q.z; w = q.w;
            return *this;
        }

        Quaternion operator+(const Quaternion& q) const { return Quaternion(x + q.x, y + q.y, z + q.z, w + q.w); }
        Quaternion operator-(const Quaternion& q) const { return Quaternion(x - q.x, y - q.y, z - q.z, w - q.w); }
        Quaternion operator*(float scalar) const { return Quaternion(x * scalar, y * scalar, z * scalar, w * scalar); }
        Quaternion operator*(const Quaternion& q) const {
            return Quaternion(
                w * q.x + x * q.w + y * q.z - z * q.y,
                w * q.y - x * q.z + y * q.w + z * q.x,
                w * q.z + x * q.y - y * q.x + z * q.w,
                w * q.w - x * q.x - y * q.y - z * q.z
            );
        }

        bool operator==(const Quaternion& q) const {
            return approxEqual(x, q.x) && approxEqual(y, q.y) && approxEqual(z, q.z) && approxEqual(w, q.w);
        }
        bool operator!=(const Quaternion& q) const { return !(*this == q); }

        float magnitude() const { return sqrt(x * x + y * y + z * z + w * w); }
        float magnitudeSquared() const { return x * x + y * y + z * z + w * w; }
        Quaternion normalized() const {
            float mag = magnitude();
            return mag > EPSILON ? *this * (1.0f / mag) : identity();
        }
        void normalize() { *this = normalized(); }
        Quaternion conjugate() const { return Quaternion(-x, -y, -z, w); }
        Quaternion inverse() const { return conjugate() * (1.0f / magnitudeSquared()); }
        float dot(const Quaternion& q) const { return x * q.x + y * q.y + z * q.z + w * q.w; }

        static Quaternion identity() { return Quaternion(0, 0, 0, 1); }
        static Quaternion angleAxis(float angle, const Vector3& axis) {
            float halfAngle = angle * 0.5f;
            float s = sin(halfAngle);
            Vector3 normalizedAxis = axis.normalized();
            return Quaternion(
                normalizedAxis.x * s,
                normalizedAxis.y * s,
                normalizedAxis.z * s,
                cos(halfAngle)
            );
        }
        static Quaternion euler(float x, float y, float z) {
            float cx = cos(x * 0.5f);
            float sx = sin(x * 0.5f);
            float cy = cos(y * 0.5f);
            float sy = sin(y * 0.5f);
            float cz = cos(z * 0.5f);
            float sz = sin(z * 0.5f);

            return Quaternion(
                sx * cy * cz - cx * sy * sz,
                cx * sy * cz + sx * cy * sz,
                cx * cy * sz - sx * sy * cz,
                cx * cy * cz + sx * sy * sz
            );
        }
    };


    struct Matrix2x2 {
        float m[2][2];

        Matrix2x2() {
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    m[i][j] = (i == j) ? 1.0f : 0.0f;
                }
            }
        }

        Matrix2x2(float m00, float m01, float m10, float m11) {
            m[0][0] = m00; m[0][1] = m01;
            m[1][0] = m10; m[1][1] = m11;
        }

        Matrix2x2(const std::initializer_list<std::initializer_list<float>>& list) {
            int i = 0;
            for (auto& row : list) {
                int j = 0;
                for (auto& val : row) {
                    m[i][j] = val;
                    ++j;
                }
                ++i;
            }
        }

        Matrix2x2 operator+(const Matrix2x2& other) const {
            Matrix2x2 result;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    result.m[i][j] = m[i][j] + other.m[i][j];
                }
            }
            return result;
        }

        Matrix2x2 operator-(const Matrix2x2& other) const {
            Matrix2x2 result;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    result.m[i][j] = m[i][j] - other.m[i][j];
                }
            }
            return result;
        }

        Matrix2x2 operator*(const Matrix2x2& other) const {
            Matrix2x2 result;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    result.m[i][j] = 0;
                    for (int k = 0; k < 2; ++k) {
                        result.m[i][j] += m[i][k] * other.m[k][j];
                    }
                }
            }
            return result;
        }

        Matrix2x2 operator*(float scalar) const {
            Matrix2x2 result;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    result.m[i][j] = m[i][j] * scalar;
                }
            }
            return result;
        }

        Vector2 operator*(const Vector2& v) const {
            return Vector2(
                m[0][0] * v.x + m[0][1] * v.y,
                m[1][0] * v.x + m[1][1] * v.y
            );
        }

        float determinant() const {
            return m[0][0] * m[1][1] - m[0][1] * m[1][0];
        }

        Matrix2x2 transpose() const {
            return Matrix2x2(m[0][0], m[1][0], m[0][1], m[1][1]);
        }

        Matrix2x2 inverse() const {
            float det = determinant();
            if (abs(det) < EPSILON) {
                return Matrix2x2(); 
            }
            float invDet = 1.0f / det;
            return Matrix2x2(
                m[1][1] * invDet, -m[0][1] * invDet,
                -m[1][0] * invDet, m[0][0] * invDet
            );
        }

        static Matrix2x2 identity() {
            return Matrix2x2();
        }

        static Matrix2x2 rotation(float angle) {
            float c = cos(angle);
            float s = sin(angle);
            return Matrix2x2(c, -s, s, c);
        }

        static Matrix2x2 scale(float sx, float sy) {
            return Matrix2x2(sx, 0, 0, sy);
        }
    };

    struct Matrix3x3 {
        float m[3][3];

        Matrix3x3() {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    m[i][j] = (i == j) ? 1.0f : 0.0f;
                }
            }
        }

        Matrix3x3(const std::initializer_list<std::initializer_list<float>>& list) {
            int i = 0;
            for (auto& row : list) {
                int j = 0;
                for (auto& val : row) {
                    m[i][j] = val;
                    ++j;
                }
                ++i;
            }
        }

        Matrix3x3 operator+(const Matrix3x3& other) const {
            Matrix3x3 result;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    result.m[i][j] = m[i][j] + other.m[i][j];
                }
            }
            return result;
        }

        Matrix3x3 operator-(const Matrix3x3& other) const {
            Matrix3x3 result;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    result.m[i][j] = m[i][j] - other.m[i][j];
                }
            }
            return result;
        }

        Matrix3x3 operator*(const Matrix3x3& other) const {
            Matrix3x3 result;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    result.m[i][j] = 0;
                    for (int k = 0; k < 3; ++k) {
                        result.m[i][j] += m[i][k] * other.m[k][j];
                    }
                }
            }
            return result;
        }

        Matrix3x3 operator*(float scalar) const {
            Matrix3x3 result;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    result.m[i][j] = m[i][j] * scalar;
                }
            }
            return result;
        }

        Vector3 operator*(const Vector3& v) const {
            return Vector3(
                m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
            );
        }

        float determinant() const {
            return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        }

        Matrix3x3 transpose() const {
            Matrix3x3 result;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    result.m[i][j] = m[j][i];
                }
            }
            return result;
        }

        Matrix3x3 inverse() const {
            float det = determinant();
            if (abs(det) < EPSILON) {
                return Matrix3x3(); 
            }

            float invDet = 1.0f / det;
            Matrix3x3 result;

            result.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invDet;
            result.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet;
            result.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet;
            result.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invDet;
            result.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet;
            result.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * invDet;
            result.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invDet;
            result.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * invDet;
            result.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet;

            return result;
        }

        static Matrix3x3 identity() {
            return Matrix3x3();
        }

        static Matrix3x3 translation(float x, float y) {
            Matrix3x3 result;
            result.m[0][2] = x;
            result.m[1][2] = y;
            return result;
        }

        static Matrix3x3 rotation(float angle) {
            float c = cos(angle);
            float s = sin(angle);
            return Matrix3x3({ {c, -s, 0}, {s, c, 0}, {0, 0, 1} });
        }

        static Matrix3x3 scale(float sx, float sy) {
            return Matrix3x3({ {sx, 0, 0}, {0, sy, 0}, {0, 0, 1} });
        }
    };

    struct Matrix4x4 {
        float m[4][4];

        Matrix4x4() {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    m[i][j] = (i == j) ? 1.0f : 0.0f;
                }
            }
        }

        Matrix4x4(const std::initializer_list<std::initializer_list<float>>& list) {
            int i = 0;
            for (auto& row : list) {
                int j = 0;
                for (auto& val : row) {
                    m[i][j] = val;
                    ++j;
                }
                ++i;
            }
        }

        Matrix4x4 operator+(const Matrix4x4& other) const {
            Matrix4x4 result;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    result.m[i][j] = m[i][j] + other.m[i][j];
                }
            }
            return result;
        }

        Matrix4x4 operator-(const Matrix4x4& other) const {
            Matrix4x4 result;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    result.m[i][j] = m[i][j] - other.m[i][j];
                }
            }
            return result;
        }

        Matrix4x4 operator*(const Matrix4x4& other) const {
            Matrix4x4 result;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    result.m[i][j] = 0;
                    for (int k = 0; k < 4; ++k) {
                        result.m[i][j] += m[i][k] * other.m[k][j];
                    }
                }
            }
            return result;
        }

        Matrix4x4 operator*(float scalar) const {
            Matrix4x4 result;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    result.m[i][j] = m[i][j] * scalar;
                }
            }
            return result;
        }

        Vector4 operator*(const Vector4& v) const {
            return Vector4(
                m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
                m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w
            );
        }

        Vector3 transformPoint(const Vector3& point) const {
            Vector4 result = *this * Vector4(point.x, point.y, point.z, 1.0f);
            return Vector3(result.x, result.y, result.z);
        }

        Vector3 transformDirection(const Vector3& direction) const {
            Vector4 result = *this * Vector4(direction.x, direction.y, direction.z, 0.0f);
            return Vector3(result.x, result.y, result.z);
        }

        float determinant() const {
            float det = 0;
            for (int i = 0; i < 4; ++i) {
                Matrix3x3 minor;
                for (int j = 1; j < 4; ++j) {
                    int col = 0;
                    for (int k = 0; k < 4; ++k) {
                        if (k == i) continue;
                        minor.m[j - 1][col] = m[j][k];
                        col++;
                    }
                }
                det += (i % 2 == 0 ? 1 : -1) * m[0][i] * minor.determinant();
            }
            return det;
        }

        Matrix4x4 transpose() const {
            Matrix4x4 result;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    result.m[i][j] = m[j][i];
                }
            }
            return result;
        }

        Matrix4x4 inverse() const {
            float det = determinant();
            if (abs(det) < EPSILON) {
                return Matrix4x4(); 
            }

            Matrix4x4 result;
            float invDet = 1.0f / det;

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    Matrix3x3 minor;
                    int minorRow = 0;
                    for (int row = 0; row < 4; ++row) {
                        if (row == i) continue;
                        int minorCol = 0;
                        for (int col = 0; col < 4; ++col) {
                            if (col == j) continue;
                            minor.m[minorRow][minorCol] = m[row][col];
                            minorCol++;
                        }
                        minorRow++;
                    }
                    result.m[j][i] = ((i + j) % 2 == 0 ? 1 : -1) * minor.determinant() * invDet;
                }
            }
            return result;
        }

        static Matrix4x4 identity() {
            return Matrix4x4();
        }

        static Matrix4x4 translation(float x, float y, float z) {
            Matrix4x4 result;
            result.m[0][3] = x;
            result.m[1][3] = y;
            result.m[2][3] = z;
            return result;
        }

        static Matrix4x4 translation(const Vector3& v) {
            return translation(v.x, v.y, v.z);
        }

        static Matrix4x4 rotationX(float angle) {
            float c = cos(angle);
            float s = sin(angle);
            return Matrix4x4({ {1, 0, 0, 0}, {0, c, -s, 0}, {0, s, c, 0}, {0, 0, 0, 1} });
        }

        static Matrix4x4 rotationY(float angle) {
            float c = cos(angle);
            float s = sin(angle);
            return Matrix4x4({ {c, 0, s, 0}, {0, 1, 0, 0}, {-s, 0, c, 0}, {0, 0, 0, 1} });
        }

        static Matrix4x4 rotationZ(float angle) {
            float c = cos(angle);
            float s = sin(angle);
            return Matrix4x4({ {c, -s, 0, 0}, {s, c, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1} });
        }

        static Matrix4x4 rotation(const Vector3& euler) {
            return rotationZ(euler.z) * rotationY(euler.y) * rotationX(euler.x);
        }

        static Matrix4x4 rotation(const Quaternion& q) {
            float xx = q.x * q.x;
            float yy = q.y * q.y;
            float zz = q.z * q.z;
            float xy = q.x * q.y;
            float xz = q.x * q.z;
            float yz = q.y * q.z;
            float wx = q.w * q.x;
            float wy = q.w * q.y;
            float wz = q.w * q.z;

            return Matrix4x4({ {
                {1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0},
                {2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0},
                {2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0},
                {0, 0, 0, 1}
            } });
        }

        static Matrix4x4 scale(float sx, float sy, float sz) {
            return Matrix4x4({ {sx, 0, 0, 0}, {0, sy, 0, 0}, {0, 0, sz, 0}, {0, 0, 0, 1} });
        }

        static Matrix4x4 scale(const Vector3& s) {
            return scale(s.x, s.y, s.z);
        }

        static Matrix4x4 scale(float uniform) {
            return scale(uniform, uniform, uniform);
        }

        static Matrix4x4 TRS(const Vector3& translation, const Quaternion& rotation, const Vector3& scale) {
            return Matrix4x4::translation(translation) * Matrix4x4::rotation(rotation) * Matrix4x4::scale(scale);
        }

        static Matrix4x4 perspective(float fovy, float aspect, float near, float far) {
            float tanHalfFovy = tan(fovy * 0.5f);
            Matrix4x4 result;
            result.m[0][0] = 1.0f / (aspect * tanHalfFovy);
            result.m[1][1] = 1.0f / tanHalfFovy;
            result.m[2][2] = -(far + near) / (far - near);
            result.m[2][3] = -(2.0f * far * near) / (far - near);
            result.m[3][2] = -1.0f;
            result.m[3][3] = 0.0f;
            return result;
        }

        static Matrix4x4 orthographic(float left, float right, float bottom, float top, float near, float far) {
            Matrix4x4 result;
            result.m[0][0] = 2.0f / (right - left);
            result.m[1][1] = 2.0f / (top - bottom);
            result.m[2][2] = -2.0f / (far - near);
            result.m[0][3] = -(right + left) / (right - left);
            result.m[1][3] = -(top + bottom) / (top - bottom);
            result.m[2][3] = -(far + near) / (far - near);
            return result;
        }

        static Matrix4x4 lookAt(const Vector3& eye, const Vector3& center, const Vector3& up) {
            Vector3 f = (center - eye).normalized();
            Vector3 s = f.cross(up).normalized();
            Vector3 u = s.cross(f);

            Matrix4x4 result;
            result.m[0][0] = s.x;
            result.m[1][0] = s.y;
            result.m[2][0] = s.z;
            result.m[0][1] = u.x;
            result.m[1][1] = u.y;
            result.m[2][1] = u.z;
            result.m[0][2] = -f.x;
            result.m[1][2] = -f.y;
            result.m[2][2] = -f.z;
            result.m[3][0] = -s.dot(eye);
            result.m[3][1] = -u.dot(eye);
            result.m[3][2] = f.dot(eye);
            return result;
        }
    };




    inline Vector2 operator*(float scalar, const Vector2& v) { return v * scalar; }

    inline Vector3 operator*(float scalar, const Vector3& v) { return v * scalar; }

    inline Vector4 operator*(float scalar, const Vector4& v) { return v * scalar; }

    inline Quaternion operator*(float scalar, const Quaternion& q) { return q * scalar; }

    inline Matrix2x2 operator*(float scalar, const Matrix2x2& m) { return m * scalar; }

    inline Matrix3x3 operator*(float scalar, const Matrix3x3& m) { return m * scalar; }

    inline Matrix4x4 operator*(float scalar, const Matrix4x4& m) { return m * scalar; }


    inline float distance(const Vector2& a, const Vector2& b) {
        return (b - a).magnitude();
    }

    inline float distance(const Vector3& a, const Vector3& b) {
        return (b - a).magnitude();
    }

    inline float distance(const Vector4& a, const Vector4& b) {
        return (b - a).magnitude();
    }


    inline Vector2 lerp(const Vector2& a, const Vector2& b, float t) {
        return a.lerp(b, t);
    }

    inline Vector3 lerp(const Vector3& a, const Vector3& b, float t) {
        return a.lerp(b, t);
    }

    inline Vector4 lerp(const Vector4& a, const Vector4& b, float t) {
        return a.lerp(b, t);
    }

    inline Quaternion slerp(const Quaternion& a, const Quaternion& b, float t) {
        float dot = a.dot(b);


        Quaternion b_adjusted = (dot < 0.0f) ? Quaternion(-b.x, -b.y, -b.z, -b.w) : b;
        dot = abs(dot);


        if (dot > 0.9995f) {
            Quaternion result = a + (b_adjusted - a) * t;
            return result.normalized();
        }

        float theta_0 = acos(dot);
        float theta = theta_0 * t;
        float sin_theta = sin(theta);
        float sin_theta_0 = sin(theta_0);

        float s0 = cos(theta) - dot * sin_theta / sin_theta_0;
        float s1 = sin_theta / sin_theta_0;

        return (a * s0) + (b_adjusted * s1);
    }


    inline float clamp(float value, float min, float max) {
        return EMin(EMax(value, min), max);
    }

    inline float clamp01(float value) {
        return clamp(value, 0.0f, 1.0f);
    }


    inline float normalizeAngle(float angle) {
        while (angle > PI) angle -= TWO_PI;
        while (angle < -PI) angle += TWO_PI;
        return angle;
    }


    inline Vector3 quaternionToEuler(const Quaternion& q) {
        Vector3 euler;


        float sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
        float cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
        euler.x = atan2(sinr_cosp, cosr_cosp);


        float sinp = 2 * (q.w * q.y - q.z * q.x);
        if (abs(sinp) >= 1)
            euler.y = copysign(PI / 2, sinp); 
        else
            euler.y = asin(sinp);

        float siny_cosp = 2 * (q.w * q.z + q.x * q.y);
        float cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
        euler.z = atan2(siny_cosp, cosy_cosp);

        return euler;
    }

} // namespace EngineMath


// Ejecutar programa: Ctrl + F5 o menú Depurar > Iniciar sin depurar
// Depurar programa: F5 o menú Depurar > Iniciar depuración

// Sugerencias para primeros pasos: 1. Use la ventana del Explorador de soluciones para agregar y administrar archivos
//   2. Use la ventana de Team Explorer para conectar con el control de código fuente
//   3. Use la ventana de salida para ver la salida de compilación y otros mensajes
//   4. Use la ventana Lista de errores para ver los errores
//   5. Vaya a Proyecto > Agregar nuevo elemento para crear nuevos archivos de código, o a Proyecto > Agregar elemento existente para agregar archivos de código existentes al proyecto
//   6. En el futuro, para volver a abrir este proyecto, vaya a Archivo > Abrir > Proyecto y seleccione el archivo .sln
